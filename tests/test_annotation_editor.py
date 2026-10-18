"""
Unit tests for the interactive AnnotationEditor GUI and review card rendering.
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from target_scorer.data.autolabel import render_review_card
from target_scorer.data.schema import AnnotatedShot, TargetAnnotation
from target_scorer.gui.annotation_editor import AnnotationEditor


class TestAnnotationEditor(unittest.TestCase):
    """Headless tests for AnnotationEditor logic and persistence."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_p = Path(self.tmp_dir)

        # Create dummy canonical image
        self.canvas_size = 1024
        self.canonical_p = self.tmp_p / "test_target_canonical.png"
        dummy_img = np.full((self.canvas_size, self.canvas_size, 3), 230, dtype=np.uint8)
        import cv2
        cv2.circle(dummy_img, (512, 512), 155, (20, 20, 20), -1)  # B-3 black center
        cv2.imwrite(str(self.canonical_p), dummy_img)

        # Create sample JSON annotation
        self.shots = [
            AnnotatedShot(id=1, center_x=512.0, center_y=512.0, score=10, is_x=True),
            AnnotatedShot(id=2, center_x=525.0, center_y=520.0, score=10, is_x=False),
        ]
        self.ann = TargetAnnotation(
            target_id="test_target",
            image_filename=self.canonical_p.name,
            target_type="B-3",
            bullseye_center=(512.0, 512.0),
            shots=self.shots,
            canvas_size=self.canvas_size,
            pixels_per_mm=4.0,
            metadata={"total_score": 20, "x_count": 1}
        )
        self.json_p = self.tmp_p / "test_target.json"
        self.ann.to_json(self.json_p)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_render_review_card_standalone(self):
        import cv2
        dummy_bgr = cv2.imread(str(self.canonical_p))
        out_rev = self.tmp_p / "test_target_review.png"
        res_card = render_review_card(
            canonical_bgr=dummy_bgr,
            annotation=self.ann,
            output_path=out_rev,
            redact_pii=False,
        )
        self.assertTrue(out_rev.exists())
        self.assertEqual(res_card.shape, (self.canvas_size, self.canvas_size, 3))

    def test_gui_load_add_move_save(self):
        """Tests full cycle: load JSON, add shot, nudge, re-score, undo, save."""
        app = AnnotationEditor(target_dir_or_file=self.tmp_dir)
        try:
            self.assertEqual(len(app.json_files), 1)
            self.assertEqual(len(app.shots), 2)
            self.assertEqual(app.group_metrics.total_score, 20)
            self.assertEqual(app.group_metrics.x_count, 2)

            # 1. Add a new shot at (550, 512)
            app.push_undo()
            app.shots.append({
                "id": 3,
                "x": 550.0,
                "y": 512.0,
                "score": 0,
                "is_x": False,
                "caliber_mm": 5.69,
                "confidence": 1.0,
            })
            app.recalculate_scores()
            self.assertEqual(len(app.shots), 3)
            # Shot at 550px is ~38px from center (9.5mm), well inside B-3 10-ring (22.86mm) -> 10
            self.assertEqual(app.shots[2]["score"], 10)
            self.assertEqual(app.group_metrics.total_score, 30)

            # 2. Test Undo
            app.on_undo()
            self.assertEqual(len(app.shots), 2)
            self.assertEqual(app.group_metrics.total_score, 20)

            # 3. Add flyer shot and Save
            app.shots.append({
                "id": 3,
                "x": 200.0,
                "y": 512.0,
                "score": 0,
                "is_x": False,
                "caliber_mm": 5.69,
                "confidence": 1.0,
            })
            app.recalculate_scores()
            app.on_save()

            # Verify saved JSON on disk
            saved_ann = TargetAnnotation.from_json(self.json_p)
            self.assertEqual(len(saved_ann.shots), 3)
            self.assertTrue(saved_ann.metadata.get("manually_verified"))

            # Verify review image was regenerated on disk
            rev_p = self.tmp_p / "test_target_review.png"
            self.assertTrue(rev_p.exists())

        finally:
            app.destroy()

    def test_combobox_theme_styling(self):
        """Verifies Caliber combobox has high-contrast styling matching Open Directory button."""
        app = AnnotationEditor(target_dir_or_file=self.tmp_dir)
        try:
            # Check TCombobox configuration
            field_bg = app.style.lookup("TCombobox", "fieldbackground")
            bg = app.style.lookup("TCombobox", "background")
            fg = app.style.lookup("TCombobox", "foreground")
            self.assertEqual(field_bg, "#323440")
            self.assertEqual(bg, "#323440")
            self.assertEqual(fg, "#f0f2f5")
        finally:
            app.destroy()

    def test_delete_all_shots_with_confirmation(self):
        """Verifies Delete All Shots prompts user, clears shots, and is reversible via Undo."""
        from unittest.mock import patch

        app = AnnotationEditor(target_dir_or_file=self.tmp_dir)
        try:
            self.assertEqual(len(app.shots), 2)

            # 1. User cancels confirmation dialog -> shots remain untouched
            with patch("tkinter.messagebox.askyesno", return_value=False):
                app.on_clear_shots()
            self.assertEqual(len(app.shots), 2)

            # 2. User confirms deletion -> all shots removed
            with patch("tkinter.messagebox.askyesno", return_value=True):
                app.on_clear_shots()
            self.assertEqual(len(app.shots), 0)
            self.assertEqual(app.group_metrics.total_score, 0)
            self.assertIsNone(app.selected_shot_idx)

            # 3. User hits Undo -> restored
            app.on_undo()
            self.assertEqual(len(app.shots), 2)
            self.assertEqual(app.group_metrics.total_score, 20)
        finally:
            app.destroy()

    def test_eagle_eye_loupe_inspection_and_nudge(self):
        """Verifies Eagle Eye scoring gauge popup initialization, ring margins, and live nudging."""
        from target_scorer.gui.annotation_editor import EagleEyeLoupe

        app = AnnotationEditor(target_dir_or_file=self.tmp_dir)
        try:
            # Select shot #1 (at 512, 512 - dead center of bullseye)
            app.selected_shot_idx = 0
            app.on_open_eagle_eye()

            loupe = app.eagle_eye_dialog
            self.assertIsNotNone(loupe)
            self.assertTrue(loupe.winfo_exists())
            self.assertEqual(loupe.shot_idx, 0)
            self.assertEqual(loupe.zoom_mag, 8.0)

            # Verify awarded score readout
            shot0 = loupe.get_current_shot()
            self.assertIsNotNone(shot0)
            self.assertEqual(shot0["score"], 10)
            self.assertTrue(shot0["is_x"])
            self.assertIn("AWARDED: 10 [X]", loupe.lbl_score.cget("text"))

            # Test nudging inside loupe
            orig_x, orig_y = shot0["x"], shot0["y"]
            loupe.nudge(1.5, -1.0)
            self.assertAlmostEqual(shot0["x"], orig_x + 1.5, places=2)
            self.assertAlmostEqual(shot0["y"], orig_y - 1.0, places=2)

            # Test zoom magnification
            loupe.set_zoom(12.0)
            self.assertEqual(loupe.zoom_mag, 12.0)

            # Test cycle to next shot
            loupe.next_shot()
            self.assertEqual(loupe.shot_idx, 1)

            # Close loupe
            loupe.destroy()
            self.assertFalse(loupe.winfo_exists())
        finally:
            app.destroy()


if __name__ == "__main__":
    unittest.main()
