"""
Format-independent image canonicalization and ingestion layer.

Handles cross-platform inputs across Apple iOS (HEIC, ProRAW DNG), Android (Ultra HDR, JPEG),
desktop platforms (PNG, TIFF, BMP, WebP), raw edge memory buffers (Picamera2 BGR/RGB arrays),
and byte streams with automatic EXIF orientation transposition and sRGB color profile mapping.
"""

import io
import logging
import os
import sys
from pathlib import Path
from typing import BinaryIO, Optional, Union

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps

logger = logging.getLogger(__name__)

# Transparently register pillow_heif if available
HEIF_AVAILABLE = False
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_AVAILABLE = True
except ImportError:
    logger.debug("pillow_heif not installed; Apple HEIC/HEIF decoding unavailable.")


class ImageCanonicalizer:
    """
    Standardizes disparate image formats, color spaces, and memory representations
    into uniform, contiguous (H, W, 3) uint8 NumPy arrays.
    """

    def __init__(self, target_srgb: bool = True):
        self.target_srgb = target_srgb
        try:
            self._srgb_profile = ImageCms.createProfile("sRGB")
        except Exception as e:
            logger.warning("Failed to initialize ImageCms sRGB profile: %s. Using standard conversion.", e)
            self._srgb_profile = None

    def _normalize_color_profile(self, img: Image.Image) -> Image.Image:
        """
        Converts embedded color profiles (e.g., Apple Display P3, Adobe RGB)
        into standardized sRGB color space to prevent gamut and threshold clipping.
        """
        icc_profile = img.info.get("icc_profile")
        if icc_profile and self.target_srgb and self._srgb_profile is not None:
            try:
                input_profile = ImageCms.getOpenProfile(io.BytesIO(icc_profile))
                img = ImageCms.profileToProfile(
                    img,
                    input_profile,
                    self._srgb_profile,
                    outputMode="RGB"
                )
            except Exception as e:
                logger.debug("CMS color profile conversion fallback: %s", e)
                img = img.convert("RGB")
        else:
            if img.mode != "RGB":
                img = img.convert("RGB")
        return img

    def _process_pil_image(self, img: Image.Image) -> np.ndarray:
        """
        Applies EXIF orientation correction and color management to a PIL Image,
        returning a clean (H, W, 3) RGB uint8 NumPy array.
        """
        # 1. Correct orientation via EXIF metadata (tags 1 through 8)
        try:
            img = ImageOps.exif_transpose(img)
        except Exception as e:
            logger.debug("EXIF transposition skipped: %s", e)

        # 2. Normalize 16-bit or floating-point depth before RGB conversion
        # Prevents Pillow's default convert('RGB') from clamping values > 255 to white
        if img.mode in ("I;16", "I;16B", "I;16L", "I;16N", "I", "F"):
            arr = np.array(img)
            if np.issubdtype(arr.dtype, np.floating):
                if arr.max() <= 1.0:
                    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    arr = arr.clip(0, 255).astype(np.uint8)
            elif arr.dtype == np.uint16 or arr.max() > 255:
                arr = (arr / 256.0).clip(0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr, mode="L")

        # 3. Convert color space and enforce 3-channel RGB
        img = self._normalize_color_profile(img)

        # 4. Convert to contiguous uint8 NumPy array
        rgb_array = np.ascontiguousarray(img, dtype=np.uint8)
        return rgb_array

    def _process_ndarray(self, arr: np.ndarray, color_order: str = "BGR") -> np.ndarray:
        """
        Ingests a raw in-memory array (e.g., from Picamera2 or video frame grabber),
        validates dimensions and dtypes, and returns an (H, W, 3) BGR array.
        """
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(arr)}")

        # Handle floating-point or 16-bit integer image buffers
        if np.issubdtype(arr.dtype, np.floating):
            if arr.max() <= 1.0:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
        elif arr.dtype == np.uint16 or arr.max() > 255:
            arr = (arr / 256.0).clip(0, 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        # Dimension normalization
        if arr.ndim == 2:  # Grayscale (H, W)
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

        if arr.ndim == 3:
            h, w, c = arr.shape
            if c == 1:
                return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            elif c == 4:
                # Strip alpha channel
                if color_order.upper() == "RGB":
                    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
                else:
                    return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            elif c == 3:
                if color_order.upper() == "RGB":
                    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                else:
                    return np.ascontiguousarray(arr)
            else:
                raise ValueError(f"Unsupported channel dimension in array: {arr.shape}")

        raise ValueError(f"Unsupported ndarray dimensions: {arr.ndim} (expected 2 or 3)")

    def load_to_bgr(
        self,
        source: Union[str, Path, bytes, io.BytesIO, BinaryIO, np.ndarray],
        color_order: str = "BGR",
    ) -> np.ndarray:
        """
        Canonical loader producing an (H, W, 3) uint8 BGR image for OpenCV operations.

        Parameters
        ----------
        source : Union[str, Path, bytes, io.BytesIO, BinaryIO, np.ndarray]
            Image source. Can be a filesystem path, '-' for stdin, raw bytes,
            a byte stream, or an existing in-memory NumPy array.
        color_order : str, default="BGR"
            For raw NumPy arrays, specifies whether the input channels are 'BGR' or 'RGB'.
            Ignored for files, streams, or bytes where format headers dictate color.

        Returns
        -------
        np.ndarray
            Contiguous (H, W, 3) uint8 BGR image array.
        """
        # Case 1: In-memory array (Picamera2, camera frame, OpenCV buffer)
        if isinstance(source, np.ndarray):
            return self._process_ndarray(source, color_order=color_order)

        # Case 2: Standard input stream ('-')
        if isinstance(source, (str, Path)) and str(source) == "-":
            data = sys.stdin.buffer.read()
            return self.load_to_bgr(data)

        # Case 3: In-memory byte stream or raw bytes
        if isinstance(source, (bytes, io.BytesIO, BinaryIO)) or hasattr(source, "read"):
            stream = io.BytesIO(source) if isinstance(source, bytes) else source
            try:
                with Image.open(stream) as img:
                    rgb_array = self._process_pil_image(img)
                    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            except Exception as e:
                # Fallback to cv2.imdecode if PIL fails
                if isinstance(source, bytes):
                    buf = np.frombuffer(source, dtype=np.uint8)
                    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    if decoded is not None:
                        return decoded
                raise ValueError(f"Failed to decode byte stream: {e}") from e

        # Case 4: File path (str or Path)
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image source not found: '{source}'")

        try:
            with Image.open(path) as img:
                rgb_array = self._process_pil_image(img)
                return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        except Exception as e:
            # Fallback to cv2.imread for specialized formats
            logger.debug("Pillow open failed for '%s' (%s), trying cv2.imread fallback", path, e)
            img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img_bgr is not None:
                return img_bgr
            if not HEIF_AVAILABLE and path.suffix.lower() in {".heic", ".heif"}:
                raise ImportError(
                    f"Cannot decode Apple HEIC file '{source}': 'pillow-heif' is not installed. "
                    "Install it via: uv pip install pillow-heif"
                ) from e
            raise ValueError(f"Failed to load image from '{source}': {e}") from e

    def load_to_rgb(
        self,
        source: Union[str, Path, bytes, io.BytesIO, BinaryIO, np.ndarray],
        color_order: str = "BGR",
    ) -> np.ndarray:
        """
        Canonical loader producing an (H, W, 3) uint8 RGB image for PyTorch / PIL operations.

        Parameters
        ----------
        source : Union[str, Path, bytes, io.BytesIO, BinaryIO, np.ndarray]
            Image source (path, bytes, stream, or NumPy array).
        color_order : str, default="BGR"
            Color channel order if source is a raw NumPy array.

        Returns
        -------
        np.ndarray
            Contiguous (H, W, 3) uint8 RGB image array.
        """
        # If input is already an ndarray with RGB, process efficiently
        if isinstance(source, np.ndarray) and color_order.upper() == "RGB":
            # If 3 channels and uint8 contiguous, normalize shape and return
            if source.ndim == 3 and source.shape[2] == 3 and source.dtype == np.uint8:
                return np.ascontiguousarray(source)

        bgr = self.load_to_bgr(source, color_order=color_order)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# Global default canonicalizer instance
_canonicalizer = ImageCanonicalizer()


def canonicalize_image(
    source: Union[str, Path, bytes, io.BytesIO, BinaryIO, np.ndarray],
    target_format: str = "BGR",
    color_order: str = "BGR",
) -> np.ndarray:
    """
    Convenience function to canonicalize any image source into a uniform NumPy array.

    Parameters
    ----------
    source : Union[str, Path, bytes, io.BytesIO, BinaryIO, np.ndarray]
        Input image from filesystem, network, memory stream, or camera buffer.
    target_format : str, default="BGR"
        Desired output channel ordering: 'BGR' (OpenCV) or 'RGB' (PyTorch/Pillow).
    color_order : str, default="BGR"
        Input channel ordering if source is an in-memory NumPy array.

    Returns
    -------
    np.ndarray
        Contiguous (H, W, 3) uint8 array in requested target_format.
    """
    if target_format.upper() == "RGB":
        return _canonicalizer.load_to_rgb(source, color_order=color_order)
    return _canonicalizer.load_to_bgr(source, color_order=color_order)
