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
from target_scorer.gui.annotation_editor import AnnotationEditor, TargetEntry


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

    def test_mixed_target_discovery(self):
        """Verifies unified discovery of labeled JSONs, unlabeled images, exclusion of review cards, and single image loading."""
        import cv2

        sub_dir = self.tmp_p / "mixed_targets"
        sub_dir.mkdir(parents=True)

        # 1. Labeled target: json + canonical + review
        t1_canon = sub_dir / "target_01_canonical.png"
        cv2.imwrite(str(t1_canon), np.full((1024, 1024, 3), 220, dtype=np.uint8))
        t1_ann = TargetAnnotation(
            target_id="target_01",
            image_filename=t1_canon.name,
            target_type="B-2",
            bullseye_center=(512.0, 512.0),
            shots=[AnnotatedShot(id=1, center_x=512.0, center_y=512.0, score=10)],
            canvas_size=1024,
            pixels_per_mm=4.0,
        )
        t1_json = sub_dir / "target_01.json"
        t1_ann.to_json(t1_json)
        t1_rev = sub_dir / "target_01_review.png"
        cv2.imwrite(str(t1_rev), np.full((1024, 1024, 3), 200, dtype=np.uint8))

        # 2. Unlabeled raw target photo
        t2_raw = sub_dir / "target_02_raw.jpg"
        cv2.imwrite(str(t2_raw), np.full((800, 600, 3), 210, dtype=np.uint8))

        # 3. Unlabeled standalone canonical image
        t3_canon = sub_dir / "target_03_canonical.png"
        cv2.imwrite(str(t3_canon), np.full((1024, 1024, 3), 220, dtype=np.uint8))

        # 4. Stray review card - MUST BE EXCLUDED from discovery!
        stray_rev = sub_dir / "stray_review.png"
        cv2.imwrite(str(stray_rev), np.full((1024, 1024, 3), 150, dtype=np.uint8))

        app = AnnotationEditor(target_dir_or_file=sub_dir)
        try:
            # Check target discovery count (exactly 3 valid targets: target_01, target_02, target_03)
            self.assertEqual(len(app.target_entries), 3)

            # Check that stray review card was excluded
            discovered_ids = [e.target_id for e in app.target_entries]
            self.assertIn("target_01", discovered_ids)
            self.assertIn("target_02", discovered_ids)
            self.assertIn("target_03", discovered_ids)
            self.assertNotIn("stray", discovered_ids)
            self.assertNotIn("stray_review", discovered_ids)

            # Check labeled status
            t1_entry = next(e for e in app.target_entries if e.target_id == "target_01")
            self.assertTrue(t1_entry.is_labeled)
            self.assertEqual(t1_entry.json_path, t1_json)

            t2_entry = next(e for e in app.target_entries if e.target_id == "target_02")
            self.assertFalse(t2_entry.is_labeled)
            self.assertIsNone(t2_entry.json_path)
            self.assertEqual(t2_entry.image_path, t2_raw)

            t3_entry = next(e for e in app.target_entries if e.target_id == "target_03")
            self.assertFalse(t3_entry.is_labeled)
            self.assertIsNone(t3_entry.json_path)
            self.assertEqual(t3_entry.canonical_path, t3_canon)

            # Backwards compatibility: json_files only includes valid JSONs
            self.assertEqual(len(app.json_files), 1)
        finally:
            app.destroy()

        # 5. Single unlabeled raw image loading
        app_single = AnnotationEditor(target_dir_or_file=t2_raw)
        try:
            self.assertEqual(len(app_single.target_entries), 1)
            self.assertFalse(app_single.target_entries[0].is_labeled)
            self.assertFalse(app_single.is_rectified)
            self.assertEqual(len(app_single.shots), 0)
        finally:
            app_single.destroy()

    def test_on_autolabel_current_and_undo_and_save(self):
        """Verifies in-editor auto-labeling, undo reversibility, and on_save disk persistence."""
        import cv2
        from unittest.mock import MagicMock

        unlabeled_p = self.tmp_p / "target_unlabeled_01.jpg"
        cv2.imwrite(str(unlabeled_p), np.full((800, 800, 3), 220, dtype=np.uint8))

        app = AnnotationEditor(target_dir_or_file=unlabeled_p, target_type_override="B-2")
        try:
            self.assertFalse(app.is_rectified)
            self.assertEqual(len(app.shots), 0)

            # Mock pipeline predict returning 2 detected shots
            mock_shots = [
                AnnotatedShot(id=1, center_x=512.0, center_y=512.0, score=10, is_x=True, caliber_mm=5.69, confidence=0.98),
                AnnotatedShot(id=2, center_x=570.0, center_y=512.0, score=9, is_x=False, caliber_mm=5.69, confidence=0.95),
            ]
            mock_ann = TargetAnnotation(
                target_id="target_unlabeled_01",
                image_filename="target_unlabeled_01_canonical.png",
                target_type="B-2",
                bullseye_center=(512.0, 512.0),
                shots=mock_shots,
                canvas_size=1024,
                pixels_per_mm=4.0,
                metadata={"autolabeled": True},
            )
            mock_canon = np.full((1024, 1024, 3), 230, dtype=np.uint8)

            mock_pipeline = MagicMock()
            mock_pipeline.predict.return_value = (mock_ann, mock_canon)
            app._get_autolabel_pipeline = MagicMock(return_value=mock_pipeline)

            # 1. Run Auto-label Current
            app.on_autolabel_current()
            self.assertTrue(app.is_rectified)
            self.assertEqual(len(app.shots), 2)
            self.assertEqual(app.shots[0]["score"], 10)
            self.assertFalse(app.shots[0]["is_x"])  # B-2 has no X-ring per NRA Rule 4.2
            self.assertEqual(app.shots[1]["score"], 9)

            # 2. Test Undo reverts to empty raw state
            app.on_undo()
            self.assertEqual(len(app.shots), 0)

            # 3. Re-apply auto-label and Save
            app.on_autolabel_current()
            self.assertEqual(len(app.shots), 2)
            app.on_save()

            # Verify files written to disk
            saved_json = self.tmp_p / "target_unlabeled_01.json"
            saved_canon = self.tmp_p / "target_unlabeled_01_canonical.png"
            saved_rev = self.tmp_p / "target_unlabeled_01_review.png"

            self.assertTrue(saved_json.exists())
            self.assertTrue(saved_canon.exists())
            self.assertTrue(saved_rev.exists())

            # Verify TargetEntry state updated
            self.assertTrue(app.current_entry.is_labeled)
            self.assertEqual(app.current_entry.json_path, saved_json)
            self.assertEqual(app.current_entry.canonical_path, saved_canon)

            # Verify deserialized annotation
            restored = TargetAnnotation.from_json(saved_json)
            self.assertEqual(len(restored.shots), 2)
            self.assertTrue(restored.metadata.get("manually_verified"))
        finally:
            app.destroy()

    def test_on_rectify_manual(self):
        """Verifies manual rectification without neural inference, shot placement, and saving."""
        import cv2
        from unittest.mock import MagicMock, patch

        unlabeled_p = self.tmp_p / "target_manual_01.jpg"
        cv2.imwrite(str(unlabeled_p), np.full((800, 800, 3), 220, dtype=np.uint8))

        app = AnnotationEditor(target_dir_or_file=unlabeled_p, target_type_override="B-2")
        try:
            self.assertFalse(app.is_rectified)

            # Mock TargetRectifier.rectify to return canonical 1024x1024 image
            mock_rect_result = MagicMock()
            mock_rect_result.canonical_crop = np.full((1024, 1024, 3), 220, dtype=np.uint8)
            mock_rect_result.tilt_angle_deg = 5.2
            mock_rect_result.blur_metric = 145.0

            with patch("target_scorer.gui.annotation_editor.TargetRectifier.rectify", return_value=mock_rect_result):
                app.on_rectify_manual()

            self.assertTrue(app.is_rectified)
            self.assertEqual(len(app.shots), 0)
            self.assertEqual(app.canvas_size, 1024)
            self.assertEqual(app.canonical_cv2.shape, (1024, 1024, 3))
            self.assertEqual(app.bullseye, (512.0, 512.0))

            # Add a manual shot at (512, 512)
            app.shots.append({
                "id": 1,
                "x": 512.0,
                "y": 512.0,
                "score": 0,
                "is_x": False,
                "caliber_mm": 5.69,
                "confidence": 1.0,
            })
            app.recalculate_scores()
            self.assertEqual(app.shots[0]["score"], 10)
            self.assertFalse(app.shots[0]["is_x"])  # B-2 has no X-ring per NRA Rule 4.2

            # Save manual annotation
            app.on_save()

            saved_json = self.tmp_p / "target_manual_01.json"
            saved_canon = self.tmp_p / "target_manual_01_canonical.png"
            saved_rev = self.tmp_p / "target_manual_01_review.png"

            self.assertTrue(saved_json.exists())
            self.assertTrue(saved_canon.exists())
            self.assertTrue(saved_rev.exists())
            self.assertTrue(app.current_entry.is_labeled)
        finally:
            app.destroy()

    def test_on_autolabel_all_unlabeled_batch(self):
        """Verifies batch auto-labeling across multiple unlabeled targets in background thread."""
        import cv2
        from unittest.mock import MagicMock

        batch_dir = self.tmp_p / "batch_test"
        batch_dir.mkdir(parents=True)

        # Create 2 unlabeled targets
        u1 = batch_dir / "batch_target_01.jpg"
        u2 = batch_dir / "batch_target_02.jpg"
        cv2.imwrite(str(u1), np.full((800, 800, 3), 220, dtype=np.uint8))
        cv2.imwrite(str(u2), np.full((800, 800, 3), 220, dtype=np.uint8))

        app = AnnotationEditor(target_dir_or_file=batch_dir)
        try:
            self.assertEqual(len(app.target_entries), 2)
            self.assertFalse(app.target_entries[0].is_labeled)
            self.assertFalse(app.target_entries[1].is_labeled)

            # Mock pipeline
            mock_pipeline = MagicMock()
            mock_pipeline.ensure_model_loaded.return_value = True

            def mock_predict(image_source, target_id=None, **kwargs):
                t_id = target_id or "batch_mock"
                ann = TargetAnnotation(
                    target_id=t_id,
                    image_filename=f"{t_id}_canonical.png",
                    target_type="B-2",
                    bullseye_center=(512.0, 512.0),
                    shots=[AnnotatedShot(id=1, center_x=512.0, center_y=512.0, score=10, is_x=True)],
                    canvas_size=1024,
                    pixels_per_mm=4.0,
                )
                canon = np.full((1024, 1024, 3), 220, dtype=np.uint8)
                return ann, canon

            mock_pipeline.predict.side_effect = mock_predict
            app._get_autolabel_pipeline = MagicMock(return_value=mock_pipeline)

            # In headless unittests, queue after() callbacks from background thread
            # and dispatch them on main thread after join to adhere to Tkinter thread safety
            queued_afters = []
            app.after = lambda delay, func, *args: queued_afters.append((func, args))

            # Run batch auto-label
            app.on_autolabel_all_unlabeled()

            # Wait for background thread to finish
            if hasattr(app, "_batch_thread") and app._batch_thread is not None:
                app._batch_thread.join(timeout=10.0)

            # Dispatch queued callbacks on the main thread
            for func, args in queued_afters:
                func(*args)
            app.update()

            # Verify files on disk for both targets
            for t_id in ("batch_target_01", "batch_target_02"):
                self.assertTrue((batch_dir / f"{t_id}.json").exists())
                self.assertTrue((batch_dir / f"{t_id}_canonical.png").exists())
                self.assertTrue((batch_dir / f"{t_id}_review.png").exists())

            # Verify all entries transitioned to is_labeled = True
            for entry in app.target_entries:
                self.assertTrue(entry.is_labeled)
                self.assertIsNotNone(entry.json_path)
                self.assertIsNotNone(entry.canonical_path)
        finally:
            app.destroy()

    def test_target_type_and_filter_switching(self):
        """Verifies filtering (All / Unlabeled / Labeled) and Target Type switching."""
        import cv2

        filt_dir = self.tmp_p / "filter_test"
        filt_dir.mkdir(parents=True)

        # 1 labeled target
        l_canon = filt_dir / "target_labeled_canonical.png"
        cv2.imwrite(str(l_canon), np.full((1024, 1024, 3), 220, dtype=np.uint8))
        l_ann = TargetAnnotation(
            target_id="target_labeled",
            image_filename=l_canon.name,
            target_type="B-2",
            bullseye_center=(512.0, 512.0),
            shots=[AnnotatedShot(id=1, center_x=512.0, center_y=512.0, score=10)],
            canvas_size=1024,
            pixels_per_mm=4.0,
        )
        l_ann.to_json(filt_dir / "target_labeled.json")

        # 2 unlabeled targets
        cv2.imwrite(str(filt_dir / "target_unlab_1.jpg"), np.full((800, 800, 3), 220, dtype=np.uint8))
        cv2.imwrite(str(filt_dir / "target_unlab_2.jpg"), np.full((800, 800, 3), 220, dtype=np.uint8))

        app = AnnotationEditor(target_dir_or_file=filt_dir)
        try:
            self.assertEqual(len(app.target_entries), 3)

            # Filter: All
            app.filter_var.set("All")
            app.on_filter_changed()
            self.assertEqual(len(app.filtered_indices), 3)

            # Filter: Unlabeled
            app.filter_var.set("Unlabeled")
            app.on_filter_changed()
            self.assertEqual(len(app.filtered_indices), 2)
            self.assertFalse(app.current_entry.is_labeled)

            # Filter: Labeled
            app.filter_var.set("Labeled")
            app.on_filter_changed()
            self.assertEqual(len(app.filtered_indices), 1)
            self.assertTrue(app.current_entry.is_labeled)
            self.assertEqual(app.current_entry.target_id, "target_labeled")

            # Target Type Switch: B-2 vs B-3
            app.target_type_var.set("B-3")
            app.on_target_type_changed()
            self.assertEqual(app.target_type, "B-3")
            self.assertEqual(app.shots[0]["score"], 10)
            self.assertTrue(app.shots[0]["is_x"])  # Dead-center bullseye is 10-X on B-3

            app.target_type_var.set("B-2")
            app.on_target_type_changed()
            self.assertEqual(app.target_type, "B-2")
            self.assertEqual(app.shots[0]["score"], 10)
            self.assertFalse(app.shots[0]["is_x"])  # B-2 has no X-ring per NRA Rule 4.2
        finally:
            app.destroy()

    def test_missing_checkpoint_warning(self):
        """Verifies graceful handling (messagebox.showwarning) when model checkpoint path does not exist."""
        import cv2
        from unittest.mock import patch

        unlab_p = self.tmp_p / "chk_test.jpg"
        cv2.imwrite(str(unlab_p), np.full((800, 800, 3), 220, dtype=np.uint8))

        app = AnnotationEditor(target_dir_or_file=unlab_p)
        try:
            # Set a non-existent checkpoint path
            app.autolabel_checkpoint_var.set("/non/existent/checkpoint_weights.pt")

            # 1. on_autolabel_current() should warn instead of crashing
            with patch("tkinter.messagebox.showwarning") as mock_warn:
                app.on_autolabel_current()
                mock_warn.assert_called_once()
                self.assertIn("Checkpoint Not Found", mock_warn.call_args[0][0])

            # 2. on_autolabel_all_unlabeled() should warn instead of crashing
            with patch("tkinter.messagebox.showwarning") as mock_warn:
                app.on_autolabel_all_unlabeled()
                mock_warn.assert_called_once()
                self.assertIn("Checkpoint Not Found", mock_warn.call_args[0][0])
        finally:
            app.destroy()


if __name__ == "__main__":
    unittest.main()

