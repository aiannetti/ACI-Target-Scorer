"""
Unit test suite for ImageCanonicalizer, format-independent ingestion,
EXIF orientation transposition, and in-memory buffer handling.
"""

import io
import os
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

from target_scorer.geometry.canonical_loader import (
    HEIF_AVAILABLE,
    ImageCanonicalizer,
    canonicalize_image,
)
from target_scorer.geometry.rectifier import TargetRectifier


class TestCanonicalLoader(unittest.TestCase):
    """Tests for ImageCanonicalizer supporting multi-format and in-memory ingestion."""

    def setUp(self):
        self.canonicalizer = ImageCanonicalizer()
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_test_pattern(self, width: int = 120, height: int = 80) -> Image.Image:
        """Creates a distinct asymmetric color test pattern."""
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        arr[: height // 2, : width // 2] = [255, 0, 0]      # Top-left: Red
        arr[: height // 2, width // 2 :] = [0, 255, 0]      # Top-right: Green
        arr[height // 2 :, : width // 2] = [0, 0, 255]      # Bottom-left: Blue
        arr[height // 2 :, width // 2 :] = [255, 255, 255]  # Bottom-right: White
        return Image.fromarray(arr, mode="RGB")

    def test_standard_disk_formats(self):
        """Verifies ingestion of standard image formats (PNG, JPEG, BMP, TIFF, WebP)."""
        pattern = self._create_test_pattern()

        formats = [
            ("test.png", "PNG"),
            ("test.jpg", "JPEG"),
            ("test.bmp", "BMP"),
            ("test.tiff", "TIFF"),
            ("test.webp", "WEBP"),
        ]

        for filename, fmt in formats:
            path = os.path.join(self.temp_dir.name, filename)
            pattern.save(path, format=fmt)

            # Test BGR loader
            bgr = self.canonicalizer.load_to_bgr(path)
            self.assertIsInstance(bgr, np.ndarray)
            self.assertEqual(bgr.shape, (80, 120, 3))
            self.assertEqual(bgr.dtype, np.uint8)
            self.assertTrue(bgr.flags.c_contiguous)

            # Test RGB loader
            rgb = self.canonicalizer.load_to_rgb(path)
            self.assertEqual(rgb.shape, (80, 120, 3))
            self.assertEqual(rgb.dtype, np.uint8)

            # Cross-verify BGR and RGB channels match
            self.assertTrue(np.allclose(bgr, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)))

    def test_heic_ingestion(self):
        """Verifies Apple HEIC file decoding via pillow-heif."""
        self.assertTrue(HEIF_AVAILABLE, "pillow-heif should be installed in the environment.")
        import pillow_heif

        pattern = self._create_test_pattern()
        heic_path = os.path.join(self.temp_dir.name, "sample.heic")

        # Save as HEIC container
        heif_file = pillow_heif.from_pillow(pattern)
        heif_file.save(heic_path)
        self.assertTrue(os.path.exists(heic_path))

        # Ingest through canonicalizer
        bgr = self.canonicalizer.load_to_bgr(heic_path)
        self.assertIsInstance(bgr, np.ndarray)
        self.assertEqual(bgr.shape, (80, 120, 3))
        self.assertEqual(bgr.dtype, np.uint8)

        # Check top-left pixel is approximately Red (in BGR: [0, 0, 255])
        top_left_bgr = bgr[10, 10]
        self.assertGreater(top_left_bgr[2], 200)  # High Red
        self.assertLess(top_left_bgr[0], 50)     # Low Blue
        self.assertLess(top_left_bgr[1], 50)     # Low Green

    def test_tiff_variants(self):
        """Verifies ingestion of disparate TIFF variants (.tif, .tiff, LZW, 16-bit, RGBA, multi-page)."""
        pattern = self._create_test_pattern()

        # 1. Standard .tif and .tiff extensions
        for ext in [".tif", ".tiff", ".TIF", ".TIFF"]:
            path = os.path.join(self.temp_dir.name, f"sample{ext}")
            pattern.save(path, format="TIFF")
            bgr = self.canonicalizer.load_to_bgr(path)
            self.assertEqual(bgr.shape, (80, 120, 3))
            self.assertEqual(bgr.dtype, np.uint8)

        # 2. LZW compressed TIFF
        lzw_path = os.path.join(self.temp_dir.name, "compressed_lzw.tiff")
        pattern.save(lzw_path, format="TIFF", compression="tiff_lzw")
        bgr_lzw = self.canonicalizer.load_to_bgr(lzw_path)
        self.assertEqual(bgr_lzw.shape, (80, 120, 3))
        self.assertEqual(bgr_lzw.dtype, np.uint8)

        # 3. 16-bit grayscale TIFF (scanner / scientific camera)
        img_16bit = Image.fromarray(np.full((50, 50), 32768, dtype=np.uint16))
        path_16 = os.path.join(self.temp_dir.name, "scanner_16bit.tiff")
        img_16bit.save(path_16, format="TIFF")
        bgr_16 = self.canonicalizer.load_to_bgr(path_16)
        self.assertEqual(bgr_16.shape, (50, 50, 3))
        self.assertEqual(bgr_16.dtype, np.uint8)
        self.assertAlmostEqual(int(bgr_16[0, 0, 0]), 128, delta=1)

        # 4. Multi-page TIFF (page 0 extracted)
        page1 = Image.fromarray(np.full((40, 40, 3), 110, dtype=np.uint8))
        page2 = Image.fromarray(np.full((40, 40, 3), 220, dtype=np.uint8))
        multipage_path = os.path.join(self.temp_dir.name, "multipage.tiff")
        page1.save(multipage_path, format="TIFF", save_all=True, append_images=[page2])
        bgr_multi = self.canonicalizer.load_to_bgr(multipage_path)
        self.assertEqual(bgr_multi.shape, (40, 40, 3))
        self.assertEqual(bgr_multi[0, 0, 0], 110)

    def test_raw_numpy_buffers(self):
        """Verifies ingestion of raw memory arrays (Picamera2 / edge frame buffers)."""
        # 1. 2D Grayscale buffer
        gray_2d = np.full((64, 64), 128, dtype=np.uint8)
        bgr = self.canonicalizer.load_to_bgr(gray_2d)
        self.assertEqual(bgr.shape, (64, 64, 3))
        self.assertTrue(np.all(bgr == 128))

        # 2. 3D Single-channel Grayscale buffer (H, W, 1)
        gray_3d = np.full((64, 64, 1), 200, dtype=np.uint8)
        bgr_from_3d = self.canonicalizer.load_to_bgr(gray_3d)
        self.assertEqual(bgr_from_3d.shape, (64, 64, 3))
        self.assertTrue(np.all(bgr_from_3d == 200))

        # 3. 4-channel BGRA buffer (alpha channel stripped)
        bgra = np.zeros((40, 50, 4), dtype=np.uint8)
        bgra[:, :] = [50, 100, 150, 255]
        bgr_from_bgra = self.canonicalizer.load_to_bgr(bgra, color_order="BGR")
        self.assertEqual(bgr_from_bgra.shape, (40, 50, 3))
        self.assertTrue(np.array_equal(bgr_from_bgra[0, 0], [50, 100, 150]))

        # 4. 4-channel RGBA buffer with color_order="RGB"
        rgba = np.zeros((40, 50, 4), dtype=np.uint8)
        rgba[:, :] = [150, 100, 50, 255]  # R=150, G=100, B=50
        bgr_from_rgba = self.canonicalizer.load_to_bgr(rgba, color_order="RGB")
        self.assertEqual(bgr_from_rgba.shape, (40, 50, 3))
        self.assertTrue(np.array_equal(bgr_from_rgba[0, 0], [50, 100, 150]))

        # 5. Floating-point buffer [0.0, 1.0]
        float_buf = np.full((30, 30, 3), 0.5, dtype=np.float32)
        bgr_float = self.canonicalizer.load_to_bgr(float_buf)
        self.assertEqual(bgr_float.dtype, np.uint8)
        self.assertTrue(np.all(bgr_float >= 127) and np.all(bgr_float <= 128))

        # 6. uint16 buffer [0, 65535] (scanner / depth buffer)
        uint16_buf = np.full((30, 30, 3), 32768, dtype=np.uint16)
        bgr_uint16 = self.canonicalizer.load_to_bgr(uint16_buf)
        self.assertEqual(bgr_uint16.dtype, np.uint8)
        self.assertAlmostEqual(int(bgr_uint16[0, 0, 0]), 128, delta=1)

    def test_bytes_and_stream_ingestion(self):
        """Verifies ingestion from raw byte strings and io.BytesIO streams."""
        pattern = self._create_test_pattern()
        byte_io = io.BytesIO()
        pattern.save(byte_io, format="PNG")
        raw_bytes = byte_io.getvalue()

        # Load from bytes directly
        bgr1 = self.canonicalizer.load_to_bgr(raw_bytes)
        self.assertEqual(bgr1.shape, (80, 120, 3))

        # Load from BytesIO stream
        byte_io.seek(0)
        bgr2 = self.canonicalizer.load_to_bgr(byte_io)
        self.assertTrue(np.array_equal(bgr1, bgr2))

        # Test top-level convenience function
        bgr3 = canonicalize_image(raw_bytes, target_format="BGR")
        self.assertTrue(np.array_equal(bgr1, bgr3))

    def test_exif_orientation_handling(self):
        """Verifies that EXIF orientation metadata (tags 1-8) properly rotates the output."""
        base_img = Image.new("RGB", (100, 50), color=(255, 255, 255))
        # Draw distinctive marker at top-left (x < 20, y < 20)
        arr = np.array(base_img)
        arr[:20, :20] = [255, 0, 0]
        base_marked = Image.fromarray(arr)

        # Save with EXIF orientation 6 (indicates 90 deg counter-clockwise rotation needed)
        exif = base_marked.getexif()
        exif[0x0112] = 6  # Orientation tag
        path = os.path.join(self.temp_dir.name, "exif_rotated.jpg")
        base_marked.save(path, "JPEG", exif=exif)

        bgr = self.canonicalizer.load_to_bgr(path)
        # Orientation 6 rotates a 100x50 landscape image into a 50x100 portrait image
        self.assertEqual(bgr.shape[0], 100)
        self.assertEqual(bgr.shape[1], 50)

    def test_target_rectifier_buffer_integration(self):
        """Verifies TargetRectifier processes an in-memory buffer directly without writing to disk."""
        rectifier = TargetRectifier(target_type="B-2", canonical_dim=512, pixels_per_mm=4.0)

        # Create a synthetic target image buffer
        h, w = 600, 600
        buffer = np.full((h, w, 3), 200, dtype=np.uint8)
        # Draw aiming black bullseye
        cv2.circle(buffer, (w // 2, h // 2), 150, (30, 30, 30), -1)

        # Execute rectification directly with in-memory buffer
        result = rectifier.process_raw_image(buffer)
        self.assertEqual(result.canonical_crop.shape, (512, 512, 3))
        self.assertFalse(result.is_fallback)

        # Test rectify alias method
        result_alias = rectifier.rectify(buffer)
        self.assertEqual(result_alias.canonical_crop.shape, (512, 512, 3))
        self.assertTrue(np.array_equal(result.canonical_crop, result_alias.canonical_crop))

    def test_error_handling(self):
        """Verifies clear exception handling for invalid inputs."""
        # Non-existent file
        with self.assertRaises(FileNotFoundError):
            self.canonicalizer.load_to_bgr("non_existent_file.png")

        # Corrupted bytes
        with self.assertRaises(ValueError):
            self.canonicalizer.load_to_bgr(b"not an image file content")

        # Invalid ndarray dimension
        with self.assertRaises(ValueError):
            self.canonicalizer.load_to_bgr(np.zeros((10,), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
