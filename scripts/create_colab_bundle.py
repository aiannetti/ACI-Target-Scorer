#!/usr/bin/env python3
"""
Packages the Target Scorer repository into a clean zip bundle for Google Colab upload.
Excludes virtual environments, git history, caches, and test outputs.
"""

import os
import sys
import zipfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
OUTPUT_ZIP = os.path.join(PROJECT_ROOT, "target_scorer_colab_bundle.zip")

EXCLUDE_DIRS = {
    ".git",
    "target_env",
    "venv",
    ".venv",
    "__pycache__",
    ".idea",
    ".vscode",
    ".eggs",
    "build",
    "dist",
    "target_scorer.egg-info",
    "weights",
    "visual_inspection_reports",
    "datasets",
}

EXCLUDE_EXTS = {
    ".pyc",
    ".pyo",
    ".swp",
    ".swo",
    ".tmp",
    ".pt",
    ".pth",
}


def make_colab_bundle():
    print(f"[*] Packaging project from '{PROJECT_ROOT}'...")
    file_count = 0

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(PROJECT_ROOT):
            # Prune excluded directories in-place
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.endswith(".egg-info")]

            for file in files:
                ext = os.path.splitext(file)[1]
                if ext in EXCLUDE_EXTS or file.endswith("~") or file.endswith(".zip"):
                    continue

                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, PROJECT_ROOT)

                # Prefix with target_scorer/ inside the archive
                arcname = os.path.join("target_scorer", rel_path)
                zf.write(abs_path, arcname)
                file_count += 1

    size_mb = os.path.getsize(OUTPUT_ZIP) / (1024 * 1024)
    print(f"[✓] Created Colab archive: '{OUTPUT_ZIP}'")
    print(f"    --> Total Files: {file_count}")
    print(f"    --> Archive Size: {size_mb:.2f} MB")
    print("[*] Ready for 1-click upload to Google Colab!")


if __name__ == "__main__":
    make_colab_bundle()
