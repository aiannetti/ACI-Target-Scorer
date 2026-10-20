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
        """Verifies Caliber combobox has modern Sun Valley styling or fallback clam styling."""
        app = AnnotationEditor(target_dir_or_file=self.tmp_dir)
        try:
            if app.has_sv_ttk:
                self.assertIn("sun-valley", app.style.theme_use())
                self.assertEqual(app.current_theme, "dark")
            else:
                field_bg = app.style.lookup("TCombobox", "fieldbackground")
                bg = app.style.lookup("TCombobox", "background")
                fg = app.style.lookup("TCombobox", "foreground")
                self.assertEqual(field_bg, "#323440")
                self.assertEqual(bg, "#323440")
                self.assertEqual(fg, "#f0f2f5")
        finally:
            app.destroy()

    def test_theme_toggle_dark_and_light(self):
        """Verifies runtime toggling between Sun Valley Dark and Light themes."""
        app = AnnotationEditor(target_dir_or_file=self.tmp_dir)
        try:
            if not app.has_sv_ttk:
                self.skipTest("sv-ttk not installed in environment.")
            self.assertEqual(app.current_theme, "dark")
            self.assertEqual(app.cget("bg"), "#1c1c1c")
            self.assertIn("Light", app.btn_theme.cget("text"))

            # Toggle to Light
            app.toggle_theme()
            self.assertEqual(app.current_theme, "light")
            self.assertEqual(app.cget("bg"), "#fafafa")
            self.assertIn("Dark", app.btn_theme.cget("text"))

            # Toggle back to Dark
            app.toggle_theme()
            self.assertEqual(app.current_theme, "dark")
            self.assertEqual(app.cget("bg"), "#1c1c1c")
            self.assertIn("Light", app.btn_theme.cget("text"))
        finally:
            app.destroy()

    def test_eagle_eye_theme_sync(self):
        """Verifies EagleEyeLoupe background and reticle canvas synchronize with theme changes."""
        app = AnnotationEditor(target_dir_or_file=self.tmp_dir)
        try:
            if not app.has_sv_ttk:
                self.skipTest("sv-ttk not installed in environment.")
            app.on_open_eagle_eye()
            loupe = app.eagle_eye_dialog
            self.assertIsNotNone(loupe)
            self.assertEqual(loupe.cget("bg"), "#1c1c1c")
            self.assertEqual(loupe.canvas.cget("highlightbackground"), "#323440")

            # Toggle theme to Light while loupe is open
            app.toggle_theme()
            self.assertEqual(loupe.cget("bg"), "#fafafa")
            self.assertEqual(loupe.canvas.cget("highlightbackground"), "#d0d0d0")

            # Toggle back to Dark
            app.toggle_theme()
            self.assertEqual(loupe.cget("bg"), "#1c1c1c")
            self.assertEqual(loupe.canvas.cget("highlightbackground"), "#323440")
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
            self.assertIsNone(app.eagle_eye_dialog)
        finally:
            app.destroy()

    def test_eagle_eye_toggle_and_exit_handlers(self):
        """Verifies Eagle Eye toggling, on_close handlers, and parent reference resets."""
        app = AnnotationEditor(target_dir_or_file=self.tmp_dir)
        try:
            # 1. Open Eagle Eye
            app.on_open_eagle_eye()
            self.assertIsNotNone(app.eagle_eye_dialog)
            self.assertTrue(app.eagle_eye_dialog.winfo_exists())

            # 2. Toggle off via on_open_eagle_eye(toggle=True)
            app.on_open_eagle_eye(toggle=True)
            self.assertIsNone(app.eagle_eye_dialog)

            # 3. Open again and exit via on_close (simulating Esc / E / Close button)
            app.on_open_eagle_eye()
            loupe = app.eagle_eye_dialog
            self.assertIsNotNone(loupe)
            loupe.on_close()
            self.assertIsNone(app.eagle_eye_dialog)

            # 4. close_eagle_eye is safe to call repeatedly when already None
            app.close_eagle_eye()
            self.assertIsNone(app.eagle_eye_dialog)
        finally:
            app.destroy()

    def test_treeview_event_recursion_suppression(self):
        """Verifies that selecting tree rows and populating tables does not trigger recursion."""
        app = AnnotationEditor(target_dir_or_file=self.tmp_dir)
        try:
            # Open Eagle Eye
            app.on_open_eagle_eye()
            loupe = app.eagle_eye_dialog
            self.assertIsNotNone(loupe)

            # Manually trigger on_tree_select with same item - should return immediately without loop
            app.selected_shot_idx = 0
            app.on_tree_select()
            self.assertEqual(app.selected_shot_idx, 0)

            # Programmatic selection must suppress treeview events
            self.assertFalse(app._suppress_tree_events)
            app._select_tree_row(1)
            self.assertFalse(app._suppress_tree_events)

            # Set shot index via loupe
            loupe.set_shot_index(1)
            self.assertEqual(app.selected_shot_idx, 1)
            self.assertEqual(loupe.shot_idx, 1)

            # Repopulating shot table must not cause recursion
            app.recalculate_scores()
            self.assertFalse(app._suppress_tree_events)
        finally:
            app.destroy()

    def test_eagle_eye_cleanup_on_shot_deletion(self):
        """Verifies Eagle Eye is cleanly closed when all shots are deleted or cleared."""
        from unittest.mock import patch

        app = AnnotationEditor(target_dir_or_file=self.tmp_dir)
        try:
            app.on_open_eagle_eye()
            self.assertIsNotNone(app.eagle_eye_dialog)

            # Clear shots -> eagle eye must be automatically closed
            with patch("tkinter.messagebox.askyesno", return_value=True):
                app.on_clear_shots()
            self.assertIsNone(app.eagle_eye_dialog)
            self.assertEqual(len(app.shots), 0)
        finally:
            app.destroy()

    def test_save_with_copy_to_directory(self):
        """Verifies that enabling copy_on_save copies raw, canonical, annotation, and review to chosen directory."""
        # Create dummy raw image
        raw_dir = self.tmp_p / "raw"
        raw_dir.mkdir(parents=True)
        raw_file = raw_dir / "test_target.jpg"
        raw_file.write_bytes(b"dummy_raw_image_bytes")

        # Destination 1: Flat directory
        export_flat = self.tmp_p / "exported_flat"
        app = AnnotationEditor(target_dir_or_file=self.tmp_dir, copy_to_dir=export_flat)
        try:
            self.assertTrue(app.copy_on_save_var.get())
            self.assertEqual(app.copy_dir_var.get(), str(export_flat))

            # Modify shot
            app.shots[0]["score"] = 9
            app.on_save()

            # Verify files in export_flat
            self.assertTrue((export_flat / "test_target.json").exists())
            self.assertTrue((export_flat / "test_target_canonical.png").exists())
            self.assertTrue((export_flat / "test_target_review.png").exists())
            # Raw file copied to raw/ subfolder or export_flat
            self.assertTrue((export_flat / "raw" / "test_target.jpg").exists() or (export_flat / "test_target.jpg").exists())

            # Verify saved JSON in export directory has manual verification metadata
            exported_ann = TargetAnnotation.from_json(export_flat / "test_target.json")
            self.assertTrue(exported_ann.metadata.get("manually_verified"))
            self.assertEqual(exported_ann.shots[0].score, 9)
        finally:
            app.destroy()

        # Destination 2: Structured directory with annotations/ and canonical_1024/
        export_struct = self.tmp_p / "exported_struct"
        (export_struct / "annotations").mkdir(parents=True)
        (export_struct / "canonical_1024").mkdir(parents=True)
        app2 = AnnotationEditor(target_dir_or_file=self.tmp_dir, copy_to_dir=export_struct)
        try:
            app2.on_save()
            self.assertTrue((export_struct / "annotations" / "test_target.json").exists())
            self.assertTrue((export_struct / "canonical_1024" / "test_target_canonical.png").exists())
            self.assertTrue((export_struct / "canonical_1024" / "test_target_review.png").exists())
            self.assertTrue((export_struct / "raw" / "test_target.jpg").exists())
        finally:
            app2.destroy()


if __name__ == "__main__":
    unittest.main()

