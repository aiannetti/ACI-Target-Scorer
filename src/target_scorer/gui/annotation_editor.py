"""
Interactive GUI for manipulating shot markers, live target metrology, and JSON annotation correction.
Allows rapid point-and-click correction of missed or misidentified bullet holes on target review images.
"""

import copy
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from target_scorer.config import TargetRingsConfig
from target_scorer.data.autolabel import render_review_card
from target_scorer.data.schema import AnnotatedShot, TargetAnnotation
from target_scorer.metrology.scoring import GroupMetrics, ShotResult, score_target_group


class EagleEyeLoupe(tk.Toplevel):
    """
    Dedicated high-magnification Eagle Eye scoring gauge inspection dialog.
    Simulates physical NRA 4x-12x optical scoring loupes with an engraved caliber reticle
    and exact scoring ring arcs to inspect tangent touches and inward scoring advantage (NRA Rule 14.3).
    """

    def __init__(self, parent: "AnnotationEditor", shot_idx: int):
        super().__init__(parent)
        self.parent_editor = parent
        self.shot_idx = shot_idx
        self.zoom_mag: float = 8.0
        self.nudge_step: float = 0.5
        self.canvas_dim: int = 380
        self.loupe_photo: Optional[ImageTk.PhotoImage] = None

        self.title(f"Eagle Eye Scoring Gauge - Shot #{self.get_current_shot_id()}")
        self.geometry("450x680")
        self.minsize(420, 660)
        theme = getattr(parent, "current_theme", "dark")
        bg_color = "#1c1c1c" if theme == "dark" else "#fafafa"
        self.configure(bg=bg_color)
        self.transient(parent)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self._build_ui()
        self._bind_events()
        self.redraw()

        # Center relative to parent window without blocking event loop
        try:
            px = parent.winfo_rootx() + (parent.winfo_width() - 450) // 2
            py = parent.winfo_rooty() + (parent.winfo_height() - 680) // 2
            self.geometry(f"+{max(20, px)}+{max(20, py)}")
        except Exception:
            pass

        self.focus_force()

    def on_close(self, event=None):
        """Cleanly closes the Eagle Eye loupe dialog and returns focus to main editor."""
        try:
            if getattr(self.parent_editor, "eagle_eye_dialog", None) is self:
                self.parent_editor.eagle_eye_dialog = None
        except Exception:
            pass
        try:
            self.destroy()
        except Exception:
            pass
        try:
            self.parent_editor.focus_force()
        except Exception:
            pass

    def destroy(self):
        try:
            if getattr(self.parent_editor, "eagle_eye_dialog", None) is self:
                self.parent_editor.eagle_eye_dialog = None
        except Exception:
            pass
        super().destroy()

    def update_theme(self, theme: str):
        bg_color = "#1c1c1c" if theme == "dark" else "#fafafa"
        border_color = "#323440" if theme == "dark" else "#d0d0d0"
        self.configure(bg=bg_color)
        if hasattr(self, "canvas"):
            self.canvas.configure(highlightbackground=border_color)

    def get_current_shot(self) -> Optional[Dict[str, Any]]:
        if 0 <= self.shot_idx < len(self.parent_editor.shots):
            return self.parent_editor.shots[self.shot_idx]
        return None

    def get_current_shot_id(self) -> int:
        shot = self.get_current_shot()
        return shot["id"] if shot else 0

    def set_shot_index(self, idx: int):
        if not self.parent_editor.shots:
            self.on_close()
            return
        clamped_idx = max(0, min(len(self.parent_editor.shots) - 1, idx))
        self.shot_idx = clamped_idx
        if self.parent_editor.selected_shot_idx != clamped_idx:
            self.parent_editor.selected_shot_idx = clamped_idx
            self.parent_editor._select_tree_row(clamped_idx)
            self.parent_editor.redraw_canvas()
        shot_id = self.get_current_shot_id()
        self.title(f"Eagle Eye Scoring Gauge - Shot #{shot_id}")
        self.lbl_shot_title.config(text=f"SHOT #{shot_id} OPTICAL INSPECTION")
        self.redraw()

    def prev_shot(self):
        if not self.parent_editor.shots:
            return
        new_idx = (self.shot_idx - 1) % len(self.parent_editor.shots)
        self.set_shot_index(new_idx)

    def next_shot(self):
        if not self.parent_editor.shots:
            return
        new_idx = (self.shot_idx + 1) % len(self.parent_editor.shots)
        self.set_shot_index(new_idx)

    def set_zoom(self, mag: float):
        self.zoom_mag = mag
        self.redraw()

    def set_step(self, step: float):
        self.nudge_step = step

    def nudge(self, dx_px: float, dy_px: float):
        shot = self.get_current_shot()
        if not shot:
            return
        self.parent_editor.push_undo()
        shot["x"] = max(0.0, min(float(self.parent_editor.canvas_size), shot["x"] + dx_px))
        shot["y"] = max(0.0, min(float(self.parent_editor.canvas_size), shot["y"] + dy_px))
        self.parent_editor.recalculate_scores()
        self.parent_editor.redraw_canvas()

    def _build_ui(self):
        # 1. Top Header
        top_frame = ttk.Frame(self, padding=(12, 8))
        top_frame.pack(side=tk.TOP, fill=tk.X)

        shot_id = self.get_current_shot_id()
        self.lbl_shot_title = ttk.Label(top_frame, text=f"SHOT #{shot_id} OPTICAL INSPECTION", style="Header.TLabel")
        self.lbl_shot_title.pack(side=tk.LEFT)

        btn_close = ttk.Button(top_frame, text="✕ Close (Esc)", style="Danger.TButton", command=self.on_close)
        btn_close.pack(side=tk.RIGHT)

        # 2. Reticle Canvas Frame
        canvas_card = ttk.Frame(self, style="Card.TFrame", padding=6)
        canvas_card.pack(side=tk.TOP, padx=12, pady=(0, 6))

        theme = getattr(self.parent_editor, "current_theme", "dark")
        border_col = "#323440" if theme == "dark" else "#d0d0d0"
        self.canvas = tk.Canvas(
            canvas_card,
            width=self.canvas_dim,
            height=self.canvas_dim,
            bg="#0f1013",
            highlightthickness=1,
            highlightbackground=border_col,
            cursor="crosshair"
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # 3. Metrology Card
        metrology_card = ttk.Frame(self, style="Card.TFrame", padding=(12, 8))
        metrology_card.pack(side=tk.TOP, fill=tk.X, padx=12, pady=(0, 6))

        self.lbl_score = ttk.Label(metrology_card, text="SCORE: --", font=("Segoe UI", 13, "bold"), foreground="#00ff78")
        self.lbl_score.pack(anchor=tk.W)

        self.lbl_ruling = ttk.Label(metrology_card, text="--", font=("Segoe UI", 9, "bold"), foreground="#00d7ff")
        self.lbl_ruling.pack(anchor=tk.W, pady=(2, 2))

        self.lbl_details = ttk.Label(metrology_card, text="--", font=("Segoe UI", 8), foreground="#a0a4b0")
        self.lbl_details.pack(anchor=tk.W)

        # 4. Controls Frame
        ctrl_frame = ttk.Frame(self, padding=(12, 4))
        ctrl_frame.pack(side=tk.TOP, fill=tk.X)

        # Magnification row
        zoom_row = ttk.Frame(ctrl_frame)
        zoom_row.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))
        ttk.Label(zoom_row, text="Loupe Zoom:").pack(side=tk.LEFT, padx=(0, 6))
        for z in [6.0, 8.0, 12.0]:
            btn_z = ttk.Button(zoom_row, text=f"{int(z)}x", width=4, command=lambda val=z: self.set_zoom(val))
            btn_z.pack(side=tk.LEFT, padx=2)

        # Step size row
        step_row = ttk.Frame(ctrl_frame)
        step_row.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))
        ttk.Label(step_row, text="Nudge Step:").pack(side=tk.LEFT, padx=(0, 6))
        for st in [0.2, 0.5, 1.0]:
            btn_s = ttk.Button(step_row, text=f"{st}px", width=5, command=lambda val=st: self.set_step(val))
            btn_s.pack(side=tk.LEFT, padx=2)

        # Nudge Pad
        pad_frame = ttk.Frame(ctrl_frame)
        pad_frame.pack(side=tk.TOP, pady=(0, 6))

        btn_up = ttk.Button(pad_frame, text="▲", width=4, command=lambda: self.nudge(0, -self.nudge_step))
        btn_up.grid(row=0, column=1, padx=2, pady=2)
        btn_left = ttk.Button(pad_frame, text="◀", width=4, command=lambda: self.nudge(-self.nudge_step, 0))
        btn_left.grid(row=1, column=0, padx=2, pady=2)
        btn_center = ttk.Label(pad_frame, text="✛", font=("Segoe UI", 11, "bold"), foreground="#ffd700", anchor="center")
        btn_center.grid(row=1, column=1, padx=2, pady=2)
        btn_right = ttk.Button(pad_frame, text="▶", width=4, command=lambda: self.nudge(self.nudge_step, 0))
        btn_right.grid(row=1, column=2, padx=2, pady=2)
        btn_down = ttk.Button(pad_frame, text="▼", width=4, command=lambda: self.nudge(0, self.nudge_step))
        btn_down.grid(row=2, column=1, padx=2, pady=2)

        # Bottom Shot Navigation Bar
        nav_row = ttk.Frame(ctrl_frame)
        nav_row.pack(side=tk.TOP, fill=tk.X, pady=(4, 0))

        btn_prev = ttk.Button(nav_row, text="◀ Prev Shot (P)", command=self.prev_shot)
        btn_prev.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))

        btn_next = ttk.Button(nav_row, text="Next Shot (N) ▶", command=self.next_shot)
        btn_next.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(3, 0))

        # Big dedicated Exit button
        exit_row = ttk.Frame(ctrl_frame)
        exit_row.pack(side=tk.TOP, fill=tk.X, pady=(8, 0))

        btn_exit = ttk.Button(exit_row, text="✕ Exit Eagle Eye (Esc / E)", style="Danger.TButton", command=self.on_close)
        btn_exit.pack(fill=tk.X, expand=True)

    def _bind_events(self):
        self.bind("<Up>", lambda e: self.nudge(0, -self.nudge_step))
        self.bind("<Down>", lambda e: self.nudge(0, self.nudge_step))
        self.bind("<Left>", lambda e: self.nudge(-self.nudge_step, 0))
        self.bind("<Right>", lambda e: self.nudge(self.nudge_step, 0))

        self.bind("<Shift-Up>", lambda e: self.nudge(0, -self.nudge_step * 3))
        self.bind("<Shift-Down>", lambda e: self.nudge(0, self.nudge_step * 3))
        self.bind("<Shift-Left>", lambda e: self.nudge(-self.nudge_step * 3, 0))
        self.bind("<Shift-Right>", lambda e: self.nudge(self.nudge_step * 3, 0))

        self.bind("p", lambda e: self.prev_shot())
        self.bind("P", lambda e: self.prev_shot())
        self.bind("[", lambda e: self.prev_shot())

        self.bind("n", lambda e: self.next_shot())
        self.bind("N", lambda e: self.next_shot())
        self.bind("]", lambda e: self.next_shot())

        self.bind("e", self.on_close)
        self.bind("E", self.on_close)
        self.bind("q", self.on_close)
        self.bind("Q", self.on_close)
        self.bind("<Escape>", self.on_close)

        self.bind("+", lambda e: self.set_zoom(min(16.0, self.zoom_mag * 1.25)))
        self.bind("=", lambda e: self.set_zoom(min(16.0, self.zoom_mag * 1.25)))
        self.bind("-", lambda e: self.set_zoom(max(4.0, self.zoom_mag * 0.8)))

    def on_canvas_click(self, event):
        """Clicking inside loupe canvas centers on that point."""
        shot = self.get_current_shot()
        if not shot:
            return
        cx_loupe = self.canvas_dim / 2.0
        cy_loupe = self.canvas_dim / 2.0
        dx_can = (event.x - cx_loupe) / self.zoom_mag
        dy_can = (event.y - cy_loupe) / self.zoom_mag
        if abs(dx_can) > 0.05 or abs(dy_can) > 0.05:
            self.nudge(dx_can, dy_can)

    def redraw(self):
        self.canvas.delete("all")
        shot = self.get_current_shot()
        if not shot or self.parent_editor.canonical_pil is None:
            return

        cx_loupe = self.canvas_dim / 2.0
        cy_loupe = self.canvas_dim / 2.0
        mag = self.zoom_mag
        sx = float(shot["x"])
        sy = float(shot["y"])
        cal_mm = float(shot.get("caliber_mm", self.parent_editor.caliber_mm))
        px_per_mm = self.parent_editor.pixels_per_mm
        bx, by = self.parent_editor.bullseye

        # 1. Render magnified target crop
        iw, ih = self.parent_editor.canonical_pil.size
        half_crop = (self.canvas_dim / 2.0) / mag
        cl_x1 = max(0, int(math.floor(sx - half_crop)))
        cl_y1 = max(0, int(math.floor(sy - half_crop)))
        cl_x2 = min(iw, int(math.ceil(sx + half_crop)))
        cl_y2 = min(ih, int(math.ceil(sy + half_crop)))

        if cl_x2 > cl_x1 and cl_y2 > cl_y1:
            crop_patch = self.parent_editor.canonical_pil.crop((cl_x1, cl_y1, cl_x2, cl_y2))
            scaled_w = max(1, int(round((cl_x2 - cl_x1) * mag)))
            scaled_h = max(1, int(round((cl_y2 - cl_y1) * mag)))
            resample_mode = Image.Resampling.BILINEAR if mag <= 4.0 else Image.Resampling.NEAREST
            scaled_patch = crop_patch.resize((scaled_w, scaled_h), resample=resample_mode)
            self.loupe_photo = ImageTk.PhotoImage(scaled_patch)

            dest_x = cx_loupe + (cl_x1 - sx) * mag
            dest_y = cy_loupe + (cl_y1 - sy) * mag
            self.canvas.create_image(dest_x, dest_y, image=self.loupe_photo, anchor=tk.NW)

        # 2. Render NRA Scoring Ring Arcs passing through loupe view
        bx_loupe = cx_loupe + (bx - sx) * mag
        by_loupe = cy_loupe + (by - sy) * mag
        dist_bull_loupe = math.hypot(cx_loupe - bx_loupe, cy_loupe - by_loupe)

        rings_cfg = TargetRingsConfig(target_type=self.parent_editor.target_type)
        rings_table = rings_cfg.get_rings_table()

        for label, r_mm, _ in rings_table:
            r_ring_px = r_mm * px_per_mm * mag
            if abs(dist_bull_loupe - r_ring_px) <= (self.canvas_dim * 0.75):
                color = "#ffd700" if label == "X" else ("#00ff78" if label == "10" else "#7e8396")
                self.canvas.create_oval(
                    bx_loupe - r_ring_px, by_loupe - r_ring_px,
                    bx_loupe + r_ring_px, by_loupe + r_ring_px,
                    outline=color, width=2, dash=(4, 2) if label not in ("X", "10") else None
                )
                # Label on arc closest to loupe center
                if dist_bull_loupe > 1e-3:
                    ux = (cx_loupe - bx_loupe) / dist_bull_loupe
                    uy = (cy_loupe - by_loupe) / dist_bull_loupe
                    lx = bx_loupe + ux * r_ring_px
                    ly = by_loupe + uy * r_ring_px
                    if 15 <= lx <= self.canvas_dim - 45 and 15 <= ly <= self.canvas_dim - 15:
                        self.canvas.create_text(lx + 6, ly - 6, text=f"{label}-Ring", fill=color, font=("Segoe UI", 8, "bold"), anchor=tk.W)

        # 3. Render Caliber Flange Gauge Reticle
        rad_gauge_px = (cal_mm / 2.0) * px_per_mm * mag
        score_val = shot.get("score", 0)
        is_x = shot.get("is_x", False)
        reticle_color = "#ffd700" if is_x else ("#00ff78" if score_val == 10 else "#00e5ff")

        # Flange Gauge Circle
        self.canvas.create_oval(
            cx_loupe - rad_gauge_px, cy_loupe - rad_gauge_px,
            cx_loupe + rad_gauge_px, cy_loupe + rad_gauge_px,
            outline=reticle_color, width=2
        )
        # Engraved outer guide ring
        self.canvas.create_oval(
            cx_loupe - rad_gauge_px - 1, cy_loupe - rad_gauge_px - 1,
            cx_loupe + rad_gauge_px + 1, cy_loupe + rad_gauge_px + 1,
            outline="#ffffff", width=1, dash=(2, 4)
        )

        # Center Crosshair & Pip
        arm = 12
        self.canvas.create_line(cx_loupe - arm, cy_loupe, cx_loupe + arm, cy_loupe, fill="#ff2255", width=1)
        self.canvas.create_line(cx_loupe, cy_loupe - arm, cx_loupe, cy_loupe + arm, fill="#ff2255", width=1)
        self.canvas.create_oval(cx_loupe - 2, cy_loupe - 2, cx_loupe + 2, cy_loupe + 2, outline="#ff2255", fill="#ff2255")

        # 4. Metrology & Inward Scoring Analysis (NRA Rule 14.3)
        dist_px = math.hypot(sx - bx, sy - by)
        dist_mm = dist_px / px_per_mm
        r_bullet_mm = cal_mm / 2.0
        flange_min_dist_mm = max(0.0, dist_mm - r_bullet_mm)

        ruling_text = ""
        for label, r_mm, pts in rings_table:
            margin_mm = r_mm - flange_min_dist_mm
            if margin_mm >= 0:
                if margin_mm < 2.0:
                    ruling_text = f"Touches {label}-ring by +{margin_mm:.2f} mm (NRA Rule 14.3 Inward Break)"
                else:
                    ruling_text = f"Inside {label}-ring (+{margin_mm:.2f} mm margin to line)"
                break
        if not ruling_text:
            ruling_text = f"Outside scoring rings (Miss - {dist_mm:.1f} mm radius)"

        x_badge = " [X]" if is_x else ""
        self.lbl_score.config(text=f"AWARDED: {score_val}{x_badge}", foreground=reticle_color)
        self.lbl_ruling.config(text=ruling_text)
        self.lbl_details.config(
            text=f"Coords: ({sx:.1f}, {sy:.1f}) px  |  Radius: {dist_mm:.2f} mm  |  Caliber: {cal_mm:.2f} mm"
        )


class AnnotationEditor(tk.Tk):
    """
    High-performance interactive desktop GUI for reviewing, adding, moving, and deleting
    bullet hole markers on autolabeled target cards, with real-time NRA metrology re-scoring.
    """

    def __init__(
        self,
        target_dir_or_file: Optional[Union[str, Path]] = None,
        target_type_override: Optional[str] = None,
        copy_to_dir: Optional[Union[str, Path]] = None,
    ):
        super().__init__()

        self.title("Target Scorer - Interactive Shot Annotation & Metrology Editor")
        self.geometry("1420x920")
        self.minsize(1050, 700)

        # Apply theme styling (Sun Valley theme with fallback to classic dark)
        self.style = ttk.Style(self)
        self.current_theme = "dark"
        self.has_sv_ttk = False
        try:
            import sv_ttk
            sv_ttk.set_theme("dark")
            self.has_sv_ttk = True
        except Exception:
            self.has_sv_ttk = False

        if self.has_sv_ttk:
            self.configure(bg="#1c1c1c")
            self._configure_sv_theme()
        else:
            self.configure(bg="#18191c")
            self.style.theme_use("clam")
            self._configure_legacy_dark_theme()

        # State
        self.target_type_override = target_type_override
        self.copy_to_dir_init = Path(copy_to_dir) if copy_to_dir else None
        self.json_files: List[Path] = []
        self.current_idx: int = -1
        self.current_json_path: Optional[Path] = None
        self.current_ann: Optional[TargetAnnotation] = None

        self.canonical_cv2: Optional[np.ndarray] = None
        self.canonical_pil: Optional[Image.Image] = None
        self.photo_img: Optional[ImageTk.PhotoImage] = None

        # Editable target state
        self.shots: List[Dict[str, Any]] = []
        self.bullseye: Tuple[float, float] = (512.0, 512.0)
        self.canvas_size: int = 1024
        self.pixels_per_mm: float = 4.0
        self.target_type: str = "B-3"
        self.caliber_mm: float = 5.69
        self.group_metrics: Optional[GroupMetrics] = None

        # Canvas pan & zoom state
        self.zoom: float = 1.0
        self.pan_x: float = 512.0
        self.pan_y: float = 512.0
        self.min_zoom: float = 0.20
        self.max_zoom: float = 12.0

        # Interaction state
        self.selected_shot_idx: Optional[int] = None
        self.dragging_mode: Optional[str] = None  # 'shot', 'bullseye', 'pan'
        self.drag_start_screen: Tuple[int, int] = (0, 0)
        self.drag_start_item_pos: Tuple[float, float] = (0.0, 0.0)
        self.is_modified: bool = False
        self.undo_stack: List[List[Dict[str, Any]]] = []
        self.eagle_eye_dialog: Optional[EagleEyeLoupe] = None
        self._suppress_tree_events: bool = False

        # Modes & Display Toggles
        self.edit_mode_var = tk.StringVar(value="shot")
        self.show_rings_var = tk.BooleanVar(value=True)
        self.show_spread_var = tk.BooleanVar(value=True)
        self.show_badges_var = tk.BooleanVar(value=True)

        # Export on Save settings
        self.copy_on_save_var = tk.BooleanVar(value=bool(copy_to_dir))
        default_dest = str(copy_to_dir) if copy_to_dir else "datasets/real_targets"
        self.copy_dir_var = tk.StringVar(value=default_dest)

        self._build_ui()
        self._bind_shortcuts()

        # Initial load
        if target_dir_or_file:
            self.load_path(target_dir_or_file)
        else:
            default_dir = Path("autolabeled_photos")
            if not default_dir.exists():
                default_dir = Path("datasets/real_targets")
            if default_dir.exists():
                self.load_path(default_dir)

    def _configure_sv_theme(self):
        """Configures typography, badge highlights, and button states for Sun Valley theme."""
        theme = self.current_theme
        header_fg = "#57c8ff" if theme == "dark" else "#005fb8"
        sub_fg = "#a0a4b0" if theme == "dark" else "#5c5c5c"
        danger_fg = "#ff6b6b" if theme == "dark" else "#c62828"
        success_fg = "#00ff78" if theme == "dark" else "#0b8043"

        self.style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"), foreground=header_fg)
        self.style.configure("SubHeader.TLabel", font=("Segoe UI", 10, "bold"), foreground=sub_fg)
        self.style.configure("Danger.TButton", font=("Segoe UI", 9, "bold"), foreground=danger_fg)
        self.style.configure("Success.TButton", font=("Segoe UI", 9, "bold"), foreground=success_fg)

    def _configure_legacy_dark_theme(self):
        """Fallback theme configuration using clam for environments without sv-ttk."""
        dark_bg = "#1e1e24"
        card_bg = "#26272e"
        accent_color = "#00d7ff"
        text_fg = "#f0f2f5"

        self.style.configure(".", background=dark_bg, foreground=text_fg, font=("Segoe UI", 9))
        self.style.configure("TFrame", background=dark_bg)
        self.style.configure("Card.TFrame", background=card_bg, relief="flat")
        self.style.configure("TLabel", background=dark_bg, foreground=text_fg)
        self.style.configure("Card.TLabel", background=card_bg, foreground=text_fg)
        self.style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"), foreground=accent_color)
        self.style.configure("SubHeader.TLabel", font=("Segoe UI", 10, "bold"), foreground="#a0a4b0")

        self.style.configure("TButton", background="#323440", foreground=text_fg, borderwidth=1, padding=(8, 4))
        self.style.map("TButton", background=[("active", "#3f4252"), ("pressed", "#252733")])

        self.style.configure("Accent.TButton", background="#007acc", foreground="#ffffff", font=("Segoe UI", 9, "bold"))
        self.style.map("Accent.TButton", background=[("active", "#0098ff"), ("pressed", "#005a9e")])

        self.style.configure("Success.TButton", background="#107c41", foreground="#ffffff", font=("Segoe UI", 9, "bold"))
        self.style.map("Success.TButton", background=[("active", "#159c52"), ("pressed", "#0b582e")])

        self.style.configure("Danger.TButton", background="#a82020", foreground="#ffffff")
        self.style.map("Danger.TButton", background=[("active", "#cf2a2a"), ("pressed", "#781515")])

        self.style.configure(
            "TCombobox",
            fieldbackground="#323440",
            background="#323440",
            foreground="#f0f2f5",
            arrowcolor="#f0f2f5",
            bordercolor="#444654",
            padding=(6, 3)
        )
        self.style.map(
            "TCombobox",
            fieldbackground=[("readonly", "#323440"), ("disabled", "#2b2d38"), ("active", "#3f4252")],
            background=[("readonly", "#323440"), ("disabled", "#2b2d38"), ("active", "#3f4252")],
            foreground=[("readonly", "#f0f2f5"), ("disabled", "#70737d"), ("active", "#ffffff")],
            selectbackground=[("readonly", "#323440")],
            selectforeground=[("readonly", "#f0f2f5")]
        )
        self.option_add("*TCombobox*Listbox.background", "#323440")
        self.option_add("*TCombobox*Listbox.foreground", "#f0f2f5")
        self.option_add("*TCombobox*Listbox.selectBackground", "#007acc")
        self.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
        self.style.configure("TCheckbutton", background=dark_bg, foreground=text_fg)
        self.style.configure("TRadiobutton", background=dark_bg, foreground=text_fg)
        self.style.configure(
            "TEntry",
            fieldbackground="#2b2d38",
            background="#2b2d38",
            foreground="#f0f2f5",
            insertcolor="#f0f2f5",
            bordercolor="#444654",
            padding=(6, 3)
        )

        self.style.configure("Treeview", background="#22232a", foreground=text_fg, fieldbackground="#22232a", rowheight=24)
        self.style.map("Treeview", background=[("selected", "#005a9e")], foreground=[("selected", "#ffffff")])
        self.style.configure("Treeview.Heading", background="#2c2d36", foreground=text_fg, font=("Segoe UI", 9, "bold"))

    def toggle_theme(self):
        """Toggles between Sun Valley Dark and Light themes at runtime."""
        if not self.has_sv_ttk:
            return
        import sv_ttk
        current = sv_ttk.get_theme()
        new_theme = "light" if current == "dark" else "dark"
        sv_ttk.set_theme(new_theme)
        self._apply_theme_colors(new_theme)

    def _apply_theme_colors(self, theme: str):
        """Applies theme-specific colors across root window, labels, buttons, and sub-dialogs."""
        self.current_theme = theme
        if self.has_sv_ttk:
            bg_color = "#1c1c1c" if theme == "dark" else "#fafafa"
            header_fg = "#57c8ff" if theme == "dark" else "#005fb8"
            sub_fg = "#a0a4b0" if theme == "dark" else "#5c5c5c"
            danger_fg = "#ff6b6b" if theme == "dark" else "#c62828"
            success_fg = "#00ff78" if theme == "dark" else "#0b8043"
            status_fg = "#57c8ff" if theme == "dark" else "#005fb8"
            hint_fg = "#858999" if theme == "dark" else "#6c7080"
        else:
            bg_color = "#18191c"
            header_fg = "#00d7ff"
            sub_fg = "#a0a4b0"
            danger_fg = "#ffffff"
            success_fg = "#ffffff"
            status_fg = "#00d7ff"
            hint_fg = "#858999"

        self.configure(bg=bg_color)
        self.style.configure("Header.TLabel", foreground=header_fg)
        self.style.configure("SubHeader.TLabel", foreground=sub_fg)
        self.style.configure("Danger.TButton", foreground=danger_fg)
        self.style.configure("Success.TButton", foreground=success_fg)

        if hasattr(self, "btn_theme"):
            if theme == "dark":
                self.btn_theme.config(text="☀️ Light Mode")
            else:
                self.btn_theme.config(text="🌙 Dark Mode")

        if hasattr(self, "lbl_status_msg"):
            self.lbl_status_msg.config(foreground=status_fg)
        if hasattr(self, "lbl_hint"):
            self.lbl_hint.config(foreground=hint_fg)
        if hasattr(self, "lbl_scale_info"):
            self.lbl_scale_info.config(foreground=sub_fg)

        # Update Eagle Eye Loupe if currently open
        if getattr(self, "eagle_eye_dialog", None) and self.eagle_eye_dialog.winfo_exists():
            self.eagle_eye_dialog.update_theme(theme)

    def _build_ui(self):
        # 1. Top Control Toolbar
        top_bar = ttk.Frame(self, padding=(10, 8))
        top_bar.pack(side=tk.TOP, fill=tk.X)

        btn_open = ttk.Button(top_bar, text="📁 Open Directory...", style="Accent.TButton", command=self.on_open_directory)
        btn_open.pack(side=tk.LEFT, padx=(0, 10))

        btn_prev = ttk.Button(top_bar, text="◀ Prev (P)", command=self.on_prev_target)
        btn_prev.pack(side=tk.LEFT, padx=2)

        self.lbl_target_info = ttk.Label(top_bar, text="No targets loaded", font=("Segoe UI", 10, "bold"), width=34)
        self.lbl_target_info.pack(side=tk.LEFT, padx=6)

        btn_next = ttk.Button(top_bar, text="Next (N) ▶", command=self.on_next_target)
        btn_next.pack(side=tk.LEFT, padx=2)

        ttk.Separator(top_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # Mode Selection
        ttk.Label(top_bar, text="Mode:").pack(side=tk.LEFT, padx=(0, 4))
        rb_shot = ttk.Radiobutton(top_bar, text="Shots (Click / Drag)", variable=self.edit_mode_var, value="shot")
        rb_shot.pack(side=tk.LEFT, padx=4)
        rb_bull = ttk.Radiobutton(top_bar, text="Bullseye Center", variable=self.edit_mode_var, value="bullseye")
        rb_bull.pack(side=tk.LEFT, padx=4)

        ttk.Separator(top_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # Caliber Selector
        ttk.Label(top_bar, text="Caliber:").pack(side=tk.LEFT, padx=(0, 4))
        self.caliber_combo = ttk.Combobox(
            top_bar,
            values=[".22 LR (5.69 mm)", ".32 S&W Long (7.98 mm)", "9mm / .38 (9.02 mm)", ".45 ACP (11.48 mm)"],
            state="readonly",
            width=20
        )
        self.caliber_combo.current(0)
        self.caliber_combo.bind("<<ComboboxSelected>>", self.on_caliber_changed)
        self.caliber_combo.pack(side=tk.LEFT, padx=4)

        # Right-aligned buttons on top toolbar
        self.btn_theme = ttk.Button(
            top_bar,
            text="☀️ Light Mode" if self.current_theme == "dark" else "🌙 Dark Mode",
            command=self.toggle_theme,
            width=13
        )
        self.btn_theme.pack(side=tk.RIGHT, padx=4)

        btn_save = ttk.Button(top_bar, text="💾 Save (Ctrl+S)", style="Accent.TButton", command=self.on_save)
        btn_save.pack(side=tk.RIGHT, padx=4)

        btn_reset = ttk.Button(top_bar, text="↺ Revert", command=self.on_revert)
        btn_reset.pack(side=tk.RIGHT, padx=4)

        btn_clear = ttk.Button(top_bar, text="✕ Clear Shots", style="Danger.TButton", command=self.on_clear_shots)
        btn_clear.pack(side=tk.RIGHT, padx=4)

        # 1b. Export / Copy on Save Bar
        export_bar = ttk.Frame(self, padding=(10, 4))
        export_bar.pack(side=tk.TOP, fill=tk.X)

        cb_style = "Switch.TCheckbutton" if self.has_sv_ttk else "TCheckbutton"
        cb_copy = ttk.Checkbutton(
            export_bar,
            text="Copy on Save to directory:",
            variable=self.copy_on_save_var,
            style=cb_style
        )
        cb_copy.pack(side=tk.LEFT, padx=(0, 8))

        self.entry_copy_dir = ttk.Entry(
            export_bar,
            textvariable=self.copy_dir_var,
            width=50
        )
        self.entry_copy_dir.pack(side=tk.LEFT, padx=(0, 8))

        btn_browse_copy = ttk.Button(
            export_bar,
            text="📁 Browse Destination...",
            command=self.on_browse_copy_dir
        )
        btn_browse_copy.pack(side=tk.LEFT, padx=(0, 10))

        hint_fg = "#858999" if self.current_theme == "dark" else "#6c7080"
        self.lbl_hint = ttk.Label(
            export_bar,
            text="(copies .json, canonical image, raw photo, and review card)",
            foreground=hint_fg,
            font=("Segoe UI", 8)
        )
        self.lbl_hint.pack(side=tk.LEFT)

        # 2. Main Middle Area (Canvas + Sidebar HUD)
        middle_paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        middle_paned.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left / Center: Interactive Target Canvas
        canvas_frame = ttk.Frame(middle_paned)
        middle_paned.add(canvas_frame, weight=3)

        # Canvas toolbar (Zoom / Fit / Overlays)
        canvas_tools = ttk.Frame(canvas_frame, padding=(8, 4))
        canvas_tools.pack(side=tk.TOP, fill=tk.X)

        btn_fit = ttk.Button(canvas_tools, text="Fit (F)", command=self.on_fit_window, width=7)
        btn_fit.pack(side=tk.LEFT, padx=2)
        btn_100 = ttk.Button(canvas_tools, text="100%", command=self.on_zoom_100, width=6)
        btn_100.pack(side=tk.LEFT, padx=2)
        btn_zin = ttk.Button(canvas_tools, text="Zoom +", command=lambda: self.change_zoom(1.25), width=8)
        btn_zin.pack(side=tk.LEFT, padx=2)
        btn_zout = ttk.Button(canvas_tools, text="Zoom -", command=lambda: self.change_zoom(0.80), width=8)
        btn_zout.pack(side=tk.LEFT, padx=2)

        self.lbl_zoom = ttk.Label(canvas_tools, text="100%", font=("Segoe UI", 9, "bold"), width=7)
        self.lbl_zoom.pack(side=tk.LEFT, padx=6)

        cb_rings = ttk.Checkbutton(canvas_tools, text="NRA Ring Guides", variable=self.show_rings_var, command=self.redraw_canvas, style=cb_style)
        cb_rings.pack(side=tk.LEFT, padx=8)

        cb_spread = ttk.Checkbutton(canvas_tools, text="Extreme Spread Line", variable=self.show_spread_var, command=self.redraw_canvas, style=cb_style)
        cb_spread.pack(side=tk.LEFT, padx=8)

        cb_badges = ttk.Checkbutton(canvas_tools, text="Shot Badges", variable=self.show_badges_var, command=self.redraw_canvas, style=cb_style)
        cb_badges.pack(side=tk.LEFT, padx=8)

        # Tkinter Canvas
        self.canvas = tk.Canvas(canvas_frame, bg="#101114", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Canvas Event Bindings
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.canvas.bind("<Button-1>", self.on_canvas_left_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_left_release)

        self.canvas.bind("<Double-Button-1>", self.on_canvas_double_click)
        self.canvas.bind("<Button-3>", self.on_canvas_right_click)
        self.canvas.bind("<Button-2>", self.on_canvas_middle_click)
        self.canvas.bind("<B2-Motion>", self.on_canvas_middle_drag)
        self.canvas.bind("<Motion>", self.on_canvas_mouse_move)

        # Mouse Wheel Zooming
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)       # Windows / macOS
        self.canvas.bind("<Button-4>", lambda e: self.on_mouse_wheel_linux(e, 1))  # Linux Scroll Up
        self.canvas.bind("<Button-5>", lambda e: self.on_mouse_wheel_linux(e, -1)) # Linux Scroll Down

        # Right Side: Metrology Scorecard & Shot List HUD
        hud_frame = ttk.Frame(middle_paned, width=380, padding=(12, 10))
        middle_paned.add(hud_frame, weight=1)

        # Target Specification Card
        spec_card = ttk.Frame(hud_frame, style="Card.TFrame", padding=10)
        spec_card.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

        ttk.Label(spec_card, text="NRA TARGET SPECIFICATION", style="Header.TLabel").pack(anchor=tk.W)
        self.lbl_target_type = ttk.Label(spec_card, text="Target: NRA B-3 (50ft Timed & Rapid)", font=("Segoe UI", 10))
        self.lbl_target_type.pack(anchor=tk.W, pady=(4, 2))
        self.lbl_scale_info = ttk.Label(spec_card, text="Scale: 4.00 px/mm | Canvas: 1024x1024", foreground="#a0a4b0")
        self.lbl_scale_info.pack(anchor=tk.W)

        # Scorecard Summary Card
        score_card = ttk.Frame(hud_frame, style="Card.TFrame", padding=12)
        score_card.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

        self.lbl_total_score = ttk.Label(score_card, text="SCORE: 0 / 100", font=("Segoe UI", 16, "bold"), foreground="#00ff78")
        self.lbl_total_score.pack(anchor=tk.W)

        self.lbl_group_stats = ttk.Label(score_card, text="Shots: 0 | X-Count: 0", font=("Segoe UI", 10, "bold"))
        self.lbl_group_stats.pack(anchor=tk.W, pady=(4, 2))

        self.lbl_extreme_spread = ttk.Label(score_card, text="Extreme Spread: 0.00 mm (0.00 MOA)", foreground="#ff9100")
        self.lbl_extreme_spread.pack(anchor=tk.W)

        self.lbl_mean_radius = ttk.Label(score_card, text="Mean Radius:    0.00 mm (0.00 MOA)", foreground="#00e5ff")
        self.lbl_mean_radius.pack(anchor=tk.W)

        # Shot List Table
        table_card = ttk.Frame(hud_frame, style="Card.TFrame", padding=8)
        table_card.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        header_frame = ttk.Frame(table_card, style="Card.TFrame")
        header_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))
        ttk.Label(header_frame, text="SHOT IMPACTS", style="SubHeader.TLabel").pack(side=tk.LEFT)
        btn_renumber = ttk.Button(header_frame, text="Re-number", command=self.on_renumber_shots, width=10)
        btn_renumber.pack(side=tk.RIGHT)

        columns = ("id", "score", "x", "dist_mm", "coords")
        self.tree_shots = ttk.Treeview(table_card, columns=columns, show="headings", selectmode="browse")
        self.tree_shots.heading("id", text="#")
        self.tree_shots.heading("score", text="Score")
        self.tree_shots.heading("x", text="X?")
        self.tree_shots.heading("dist_mm", text="Radius (mm)")
        self.tree_shots.heading("coords", text="X, Y (px)")

        self.tree_shots.column("id", width=36, anchor=tk.CENTER)
        self.tree_shots.column("score", width=55, anchor=tk.CENTER)
        self.tree_shots.column("x", width=40, anchor=tk.CENTER)
        self.tree_shots.column("dist_mm", width=90, anchor=tk.E)
        self.tree_shots.column("coords", width=110, anchor=tk.CENTER)

        tree_scroll = ttk.Scrollbar(table_card, orient=tk.VERTICAL, command=self.tree_shots.yview)
        self.tree_shots.configure(yscrollcommand=tree_scroll.set)
        self.tree_shots.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree_shots.bind("<<TreeviewSelect>>", self.on_tree_select)
        self.tree_shots.bind("<Double-Button-1>", lambda e: self.on_open_eagle_eye())

        # Bottom Action Buttons on Sidebar
        side_actions = ttk.Frame(hud_frame, padding=(0, 6))
        side_actions.pack(side=tk.BOTTOM, fill=tk.X)

        row_del = ttk.Frame(side_actions)
        row_del.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))

        btn_del_shot = ttk.Button(row_del, text="Delete Shot (Del)", style="Danger.TButton", command=self.on_delete_selected)
        btn_del_shot.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))

        btn_del_all = ttk.Button(row_del, text="✕ Delete All Shots", style="Danger.TButton", command=self.on_clear_shots)
        btn_del_all.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(3, 0))

        row_tools = ttk.Frame(side_actions)
        row_tools.pack(side=tk.TOP, fill=tk.X)

        btn_eagle = ttk.Button(row_tools, text="🔍 Eagle Eye Gauge (E)", style="Accent.TButton", command=self.on_open_eagle_eye)
        btn_eagle.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))

        btn_undo = ttk.Button(row_tools, text="↺ Undo (Ctrl+Z)", command=self.on_undo)
        btn_undo.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(3, 0))

        # 3. Bottom Status Bar
        status_bar = ttk.Frame(self, padding=(10, 5))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.lbl_cursor_pos = ttk.Label(status_bar, text="Cursor: X=0.0, Y=0.0", width=30)
        self.lbl_cursor_pos.pack(side=tk.LEFT)

        status_fg = "#57c8ff" if self.current_theme == "dark" else "#005fb8"
        self.lbl_status_msg = ttk.Label(
            status_bar,
            text="Ready. Click canvas to add shots | Drag to move | Press 'E' for Eagle Eye Loupe | Ctrl+T toggles theme",
            foreground=status_fg
        )
        self.lbl_status_msg.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        self.lbl_save_status = ttk.Label(status_bar, text="", font=("Segoe UI", 9, "bold"), foreground="#00ff78")
        self.lbl_save_status.pack(side=tk.RIGHT)

    def _bind_shortcuts(self):
        self.bind("<Control-s>", lambda e: self.on_save())
        self.bind("<Control-z>", lambda e: self.on_undo())
        self.bind("<Control-t>", lambda e: self.toggle_theme())
        self.bind("<Control-T>", lambda e: self.toggle_theme())
        self.bind("<F10>", lambda e: self.toggle_theme())
        self.bind("<Delete>", lambda e: self.on_delete_selected())
        self.bind("<BackSpace>", lambda e: self.on_delete_selected())
        self.bind("e", lambda e: self.on_open_eagle_eye())
        self.bind("E", lambda e: self.on_open_eagle_eye())
        self.bind("n", lambda e: self.on_next_target())
        self.bind("N", lambda e: self.on_next_target())
        self.bind("p", lambda e: self.on_prev_target())
        self.bind("P", lambda e: self.on_prev_target())
        self.bind("f", lambda e: self.on_fit_window())
        self.bind("F", lambda e: self.on_fit_window())

        # Fine-grained arrow key coordinate nudging
        self.bind("<Up>", lambda e: self.nudge_selected_shot(0, -1))
        self.bind("<Down>", lambda e: self.nudge_selected_shot(0, 1))
        self.bind("<Left>", lambda e: self.nudge_selected_shot(-1, 0))
        self.bind("<Right>", lambda e: self.nudge_selected_shot(1, 0))
        self.bind("<Shift-Up>", lambda e: self.nudge_selected_shot(0, -5))
        self.bind("<Shift-Down>", lambda e: self.nudge_selected_shot(0, 5))
        self.bind("<Shift-Left>", lambda e: self.nudge_selected_shot(-5, 0))
        self.bind("<Shift-Right>", lambda e: self.nudge_selected_shot(5, 0))

    # =========================================================================
    # File Loading & Directory Navigation
    # =========================================================================

    def load_path(self, path: Union[str, Path]):
        p = Path(path).resolve()
        if p.is_file():
            if p.suffix.lower() == ".json":
                self.json_files = [p]
                self.current_idx = 0
            else:
                json_cand = p.with_suffix(".json")
                if json_cand.exists():
                    self.json_files = [json_cand]
                    self.current_idx = 0
        elif p.is_dir():
            # Recursively find all JSON files that match target annotation schema
            cand_files = sorted(list(p.rglob("*.json")))
            valid_files = []
            for f in cand_files:
                if f.name == "manifest.json":
                    continue
                try:
                    with open(f, "r", encoding="utf-8") as jf:
                        d = json.load(jf)
                    if "target_type" in d or "bullseye_center" in d or "shots" in d:
                        valid_files.append(f)
                except Exception:
                    continue
            self.json_files = valid_files
            self.current_idx = 0 if self.json_files else -1

        if self.json_files:
            self.load_target_index(self.current_idx)
        else:
            self.lbl_target_info.config(text="No target JSON files found.")

    def on_open_directory(self):
        dir_selected = filedialog.askdirectory(title="Select Target Directory", initialdir=os.getcwd())
        if dir_selected:
            self.load_path(dir_selected)

    def load_target_index(self, idx: int):
        if not (0 <= idx < len(self.json_files)):
            return

        self.current_idx = idx
        self.current_json_path = self.json_files[idx]
        self.lbl_target_info.config(text=f"{self.current_json_path.name} [{idx + 1} / {len(self.json_files)}]")

        # Load JSON annotation
        try:
            self.current_ann = TargetAnnotation.from_json(self.current_json_path)
        except Exception as e:
            messagebox.showerror("Error Loading JSON", f"Failed to parse '{self.current_json_path.name}':\n{e}")
            return

        self.target_type = self.target_type_override or self.current_ann.target_type
        self.canvas_size = self.current_ann.canvas_size
        self.pixels_per_mm = self.current_ann.pixels_per_mm
        self.bullseye = tuple(self.current_ann.bullseye_center)

        # Load shots into mutable state
        self.shots = []
        for s in self.current_ann.shots:
            self.shots.append({
                "id": s.id,
                "x": float(s.center_x),
                "y": float(s.center_y),
                "score": s.score,
                "is_x": s.is_x,
                "caliber_mm": s.caliber_mm,
                "confidence": s.confidence,
            })

        if self.shots:
            self.caliber_mm = self.shots[0]["caliber_mm"]
            self._update_caliber_combobox()

        # Locate corresponding canonical image
        img_path = self._resolve_canonical_image_path(self.current_json_path, self.current_ann)
        if img_path and img_path.exists():
            self.canonical_cv2 = cv2.imread(str(img_path))
            # Convert BGR to RGB for PIL
            rgb = cv2.cvtColor(self.canonical_cv2, cv2.COLOR_BGR2RGB)
            self.canonical_pil = Image.fromarray(rgb)
        else:
            # Fallback placeholder canvas
            self.canonical_cv2 = np.full((self.canvas_size, self.canvas_size, 3), 220, dtype=np.uint8)
            self.canonical_pil = Image.fromarray(self.canonical_cv2)

        self.undo_stack.clear()
        self.selected_shot_idx = None
        self.is_modified = False
        self.lbl_save_status.config(text="")

        # Recalculate metrology and fit to window
        self.recalculate_scores()
        self.on_fit_window()

        if hasattr(self, "eagle_eye_dialog") and self.eagle_eye_dialog and self.eagle_eye_dialog.winfo_exists():
            if self.shots:
                self.eagle_eye_dialog.set_shot_index(0)
            else:
                self.close_eagle_eye()

    def _resolve_canonical_image_path(self, json_p: Path, ann: TargetAnnotation) -> Optional[Path]:
        json_dir = json_p.parent
        candidates = [
            json_dir / ann.image_filename,
            json_dir / f"{ann.target_id}_canonical.png",
            json_dir / f"{ann.target_id}_review.png",
            json_dir.parent / "canonical_1024" / ann.image_filename,
            json_dir.parent / "images" / ann.image_filename,
            json_dir.parent / "canonical" / ann.image_filename,
            json_dir / f"{ann.target_id}.png",
            json_dir / f"{ann.target_id}.jpg",
            json_dir / f"{ann.target_id}.JPG",
        ]
        for c in candidates:
            if c.exists() and c.is_file():
                return c
        return None

    def _resolve_raw_image_path(self, json_p: Path, ann: TargetAnnotation) -> Optional[Path]:
        """
        Locates the original un-rectified raw photograph or scan for the target.
        Searches metadata, sibling raw/ directories, and range_photos/ archives.
        """
        # 1. Metadata attributes
        if ann.metadata:
            for key in ("raw_image", "source_image", "raw_path", "image_source"):
                raw_val = ann.metadata.get(key)
                if raw_val:
                    p = Path(raw_val)
                    if p.is_file():
                        return p
                    if (json_p.parent / p).is_file():
                        return json_p.parent / p
                    if (json_p.parent / p.name).is_file():
                        return json_p.parent / p.name

        # 2. Search common raw directory patterns
        target_id = ann.target_id
        base_stems = [target_id, target_id.replace("_canonical", "")]
        extensions = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".tif", ".tiff", ".TIFF", ".heic", ".HEIC", ".raw", ".dng"]

        search_dirs = [
            json_p.parent,
            json_p.parent / "raw",
            json_p.parent.parent / "raw",
            json_p.parent.parent.parent / "raw",
            Path("range_photos") / ann.target_type / "JPEG",
            Path("range_photos") / ann.target_type,
            Path("range_photos"),
            Path("datasets/real_targets/raw"),
            Path("datasets/calibration_targets/raw"),
        ]

        for s_dir in search_dirs:
            if s_dir.exists():
                for stem in base_stems:
                    for suffix in ("", "_raw", "_raw_photo"):
                        for ext in extensions:
                            cand = s_dir / f"{stem}{suffix}{ext}"
                            if cand.exists() and cand.is_file():
                                return cand

        # 3. Fallback: fuzzy prefix search in raw directories
        for s_dir in [json_p.parent / "raw", json_p.parent.parent / "raw", Path("range_photos")]:
            if s_dir.exists():
                for f in s_dir.rglob("*"):
                    if f.is_file() and any(f.stem.startswith(s) for s in base_stems) and f.suffix.lower() in extensions:
                        return f

        return None

    def _update_caliber_combobox(self):
        cal = self.caliber_mm
        if abs(cal - 5.69) < 0.4:
            self.caliber_combo.current(0)
        elif abs(cal - 7.98) < 0.4:
            self.caliber_combo.current(1)
        elif abs(cal - 9.02) < 0.4:
            self.caliber_combo.current(2)
        elif abs(cal - 11.48) < 0.5:
            self.caliber_combo.current(3)

    def on_caliber_changed(self, event=None):
        idx = self.caliber_combo.current()
        if idx == 0:
            self.caliber_mm = 5.69
        elif idx == 1:
            self.caliber_mm = 7.98
        elif idx == 2:
            self.caliber_mm = 9.02
        elif idx == 3:
            self.caliber_mm = 11.48

        for s in self.shots:
            s["caliber_mm"] = self.caliber_mm
        self.push_undo()
        self.recalculate_scores()
        self.redraw_canvas()

    def on_prev_target(self):
        if self.json_files and self.current_idx > 0:
            self.load_target_index(self.current_idx - 1)

    def on_next_target(self):
        if self.json_files and self.current_idx < len(self.json_files) - 1:
            self.load_target_index(self.current_idx + 1)

    # =========================================================================
    # Coordinate Transforms & Canvas Rendering
    # =========================================================================

    def img_to_screen(self, img_x: float, img_y: float) -> Tuple[float, float]:
        w_can = self.canvas.winfo_width()
        h_can = self.canvas.winfo_height()
        scr_x = (w_can / 2.0) + (img_x - self.pan_x) * self.zoom
        scr_y = (h_can / 2.0) + (img_y - self.pan_y) * self.zoom
        return scr_x, scr_y

    def screen_to_img(self, scr_x: float, scr_y: float) -> Tuple[float, float]:
        w_can = self.canvas.winfo_width()
        h_can = self.canvas.winfo_height()
        img_x = self.pan_x + (scr_x - (w_can / 2.0)) / self.zoom
        img_y = self.pan_y + (scr_y - (h_can / 2.0)) / self.zoom
        return img_x, img_y

    def on_canvas_resize(self, event):
        self.redraw_canvas()

    def on_fit_window(self):
        w_can = max(100, self.canvas.winfo_width())
        h_can = max(100, self.canvas.winfo_height())
        if self.canonical_pil is not None:
            iw, ih = self.canonical_pil.size
            self.zoom = min((w_can - 30) / float(iw), (h_can - 30) / float(ih))
            self.pan_x = iw / 2.0
            self.pan_y = ih / 2.0
        else:
            self.zoom = 0.8
            self.pan_x = self.canvas_size / 2.0
            self.pan_y = self.canvas_size / 2.0
        self.redraw_canvas()

    def on_zoom_100(self):
        self.zoom = 1.0
        self.redraw_canvas()

    def change_zoom(self, factor: float, center_screen: Optional[Tuple[int, int]] = None):
        new_zoom = max(self.min_zoom, min(self.max_zoom, self.zoom * factor))
        if abs(new_zoom - self.zoom) < 1e-4:
            return

        w_can = self.canvas.winfo_width()
        h_can = self.canvas.winfo_height()
        cx = center_screen[0] if center_screen else w_can / 2.0
        cy = center_screen[1] if center_screen else h_can / 2.0

        # Anchor image coordinate under cursor
        img_x = self.pan_x + (cx - (w_can / 2.0)) / self.zoom
        img_y = self.pan_y + (cy - (h_can / 2.0)) / self.zoom

        self.zoom = new_zoom
        self.pan_x = img_x - (cx - (w_can / 2.0)) / self.zoom
        self.pan_y = img_y - (cy - (h_can / 2.0)) / self.zoom

        self.redraw_canvas()

    def on_mouse_wheel(self, event):
        factor = 1.15 if event.delta > 0 else 0.85
        self.change_zoom(factor, (event.x, event.y))

    def on_mouse_wheel_linux(self, event, direction: int):
        factor = 1.15 if direction > 0 else 0.85
        self.change_zoom(factor, (event.x, event.y))

    def redraw_canvas(self):
        self.canvas.delete("all")
        self.lbl_zoom.config(text=f"{int(round(self.zoom * 100))}%")

        if self.canonical_pil is None:
            return

        w_can = self.canvas.winfo_width()
        h_can = self.canvas.winfo_height()
        if w_can <= 1 or h_can <= 1:
            return

        # 1. Render scaled viewport of canonical target
        iw, ih = self.canonical_pil.size
        # Image bounds visible on canvas
        left_img = max(0.0, self.pan_x - (w_can / 2.0) / self.zoom)
        top_img = max(0.0, self.pan_y - (h_can / 2.0) / self.zoom)
        right_img = min(float(iw), self.pan_x + (w_can / 2.0) / self.zoom)
        bot_img = min(float(ih), self.pan_y + (h_can / 2.0) / self.zoom)

        if right_img > left_img and bot_img > top_img:
            crop_box = (int(left_img), int(top_img), int(math.ceil(right_img)), int(math.ceil(bot_img)))
            crop_w = crop_box[2] - crop_box[0]
            crop_h = crop_box[3] - crop_box[1]
            if crop_w > 0 and crop_h > 0:
                crop = self.canonical_pil.crop(crop_box)
                scaled_w = int(round(crop_w * self.zoom))
                scaled_h = int(round(crop_h * self.zoom))
                if scaled_w > 0 and scaled_h > 0:
                    resample_mode = Image.Resampling.BILINEAR if self.zoom <= 2.5 else Image.Resampling.NEAREST
                    scaled_crop = crop.resize((scaled_w, scaled_h), resample=resample_mode)
                    self.photo_img = ImageTk.PhotoImage(scaled_crop)

                    dest_scr_x, dest_scr_y = self.img_to_screen(crop_box[0], crop_box[1])
                    self.canvas.create_image(dest_scr_x, dest_scr_y, image=self.photo_img, anchor=tk.NW)

        # 2. Draw NRA Ring Guide Contours (Optional)
        bx_scr, by_scr = self.img_to_screen(self.bullseye[0], self.bullseye[1])
        if self.show_rings_var.get():
            rings_cfg = TargetRingsConfig(target_type=self.target_type)
            for label, radius_mm, _ in rings_cfg.get_rings_table():
                rad_scr = radius_mm * self.pixels_per_mm * self.zoom
                color = "#ffd700" if label == "X" else "#555866"
                self.canvas.create_oval(
                    bx_scr - rad_scr, by_scr - rad_scr,
                    bx_scr + rad_scr, by_scr + rad_scr,
                    outline=color, width=1, dash=(3, 3) if label != "X" else None
                )

        # 3. Draw Bullseye Marker (Gold Crosshair)
        self.canvas.create_line(bx_scr - 14, by_scr, bx_scr + 14, by_scr, fill="#ffd700", width=2)
        self.canvas.create_line(bx_scr, by_scr - 14, bx_scr, by_scr + 14, fill="#ffd700", width=2)
        self.canvas.create_oval(bx_scr - 5, by_scr - 5, bx_scr + 5, by_scr + 5, outline="#ffd700", width=1)
        self.canvas.create_text(bx_scr + 16, by_scr - 10, text="Bullseye", fill="#ffd700", font=("Segoe UI", 9, "bold"), anchor=tk.W)

        # 4. Draw Extreme Spread Line (if >= 2 shots)
        if self.show_spread_var.get() and self.group_metrics and self.group_metrics.total_shots >= 2 and self.group_metrics.extreme_spread_mm > 0:
            pair = self.group_metrics.farthest_pair
            s1 = next((s for s in self.shots if s["id"] == pair[0]), None)
            s2 = next((s for s in self.shots if s["id"] == pair[1]), None)
            if s1 and s2:
                p1_scr = self.img_to_screen(s1["x"], s1["y"])
                p2_scr = self.img_to_screen(s2["x"], s2["y"])
                self.canvas.create_line(p1_scr[0], p1_scr[1], p2_scr[0], p2_scr[1], fill="#ff9100", width=2, dash=(4, 2))

        # 5. Draw Shot Markers
        for idx, shot in enumerate(self.shots):
            sx, sy = self.img_to_screen(shot["x"], shot["y"])
            rad_scr = max(2.0, (shot["caliber_mm"] / 2.0) * self.pixels_per_mm * self.zoom)

            is_selected = (idx == self.selected_shot_idx)
            is_x = shot.get("is_x", False)
            score_val = shot.get("score", 0)

            # Colors matching official HUD
            if is_x:
                gauge_color = "#ffd700"  # Gold
            elif score_val == 10:
                gauge_color = "#00ff78"  # Green
            elif score_val == 9:
                gauge_color = "#00e5ff"  # Cyan
            elif score_val >= 7:
                gauge_color = "#ff9100"  # Orange
            else:
                gauge_color = "#ff3344"  # Red

            # Caliber ring
            self.canvas.create_oval(
                sx - rad_scr, sy - rad_scr,
                sx + rad_scr, sy + rad_scr,
                outline=gauge_color, width=2
            )

            # Selection Highlight Glow
            if is_selected:
                self.canvas.create_oval(
                    sx - rad_scr - 3, sy - rad_scr - 3,
                    sx + rad_scr + 3, sy + rad_scr + 3,
                    outline="#ffffff", width=2, dash=(3, 3)
                )

            # Center crosshair marker
            arm = max(3, int(round(5 * min(2.0, self.zoom))))
            self.canvas.create_line(sx - arm, sy, sx + arm, sy, fill="#ff0044", width=1)
            self.canvas.create_line(sx, sy - arm, sx, sy + arm, fill="#ff0044", width=1)

            # Shot Badge Label
            if self.show_badges_var.get():
                tag_text = f"#{shot['id']}:{score_val}{'X' if is_x else ''}"
                bx = sx + rad_scr + 3
                by = sy - 4
                self.canvas.create_rectangle(bx - 2, by - 8, bx + 36, by + 6, fill="#1c1d22", outline=gauge_color, width=1)
                self.canvas.create_text(bx + 1, by - 1, text=tag_text, fill="#ffffff", font=("Segoe UI", 8, "bold"), anchor=tk.W)

    # =========================================================================
    # Real-Time Metrology Scoring
    # =========================================================================

    def recalculate_scores(self):
        """Calculates official NRA inward scoring for all shots in real-time."""
        rings_cfg = TargetRingsConfig(target_type=self.target_type)

        if not self.shots:
            self.group_metrics = GroupMetrics(
                total_shots=0, total_score=0, max_possible_score=0, x_count=0,
                extreme_spread_mm=0.0, extreme_spread_moa=0.0, mean_radius_mm=0.0, mean_radius_moa=0.0,
                center_of_impact_mm=(0.0, 0.0), windage_bias_mm=0.0, elevation_bias_mm=0.0,
                farthest_pair=(0, 0), target_distance_yards=16.67
            )
        else:
            peaks_tuples = [(float(s["x"]), float(s["y"]), 1.0) for s in self.shots]
            shot_results, self.group_metrics = score_target_group(
                shot_peaks=peaks_tuples,
                bullseye_peak=self.bullseye,
                pixels_per_mm=self.pixels_per_mm,
                bullet_diameter_mm=self.caliber_mm,
                rings_cfg=rings_cfg,
                distance_yards=16.67
            )

            # Update shots with computed scores and flags
            for s_res, s_dict in zip(shot_results, self.shots):
                s_dict["score"] = s_res.score_value
                s_dict["is_x"] = s_res.is_inner_x
                s_dict["dist_mm"] = s_res.radial_dist_mm

        # Update HUD labels
        self._update_hud_display()
        self._populate_shot_table()
        if hasattr(self, "eagle_eye_dialog") and self.eagle_eye_dialog and self.eagle_eye_dialog.winfo_exists():
            self.eagle_eye_dialog.redraw()

    def _update_hud_display(self):
        m = self.group_metrics
        if not m:
            return

        target_name = "NRA B-3 (50ft Timed & Rapid)" if self.target_type == "B-3" else "NRA B-2 (50ft Slow Fire)"
        self.lbl_target_type.config(text=f"Target: {target_name}")
        self.lbl_scale_info.config(text=f"Scale: {self.pixels_per_mm:.2f} px/mm | Canvas: {self.canvas_size}x{self.canvas_size}")

        max_score = len(self.shots) * 10
        score_text = f"SCORE: {m.total_score} / {max_score}"
        self.lbl_total_score.config(text=score_text)

        stats_text = f"Shots: {m.total_shots} | X-Count: {m.x_count}"
        self.lbl_group_stats.config(text=stats_text)

        es_in = m.extreme_spread_mm / 25.4
        self.lbl_extreme_spread.config(text=f"Extreme Spread: {m.extreme_spread_mm:.2f} mm ({es_in:.2f} in / {m.extreme_spread_moa:.1f} MOA)")

        mr_in = m.mean_radius_mm / 25.4
        self.lbl_mean_radius.config(text=f"Mean Radius:    {m.mean_radius_mm:.2f} mm ({mr_in:.2f} in / {m.mean_radius_moa:.1f} MOA)")

    def _populate_shot_table(self):
        self._suppress_tree_events = True
        try:
            for item in self.tree_shots.get_children():
                self.tree_shots.delete(item)

            for idx, s in enumerate(self.shots):
                dist_val = s.get("dist_mm", 0.0)
                coords_str = f"{s['x']:.1f}, {s['y']:.1f}"
                item_id = self.tree_shots.insert("", tk.END, values=(
                    f"#{s['id']}",
                    s.get("score", 0),
                    "X" if s.get("is_x") else "",
                    f"{dist_val:.2f}",
                    coords_str
                ))
                if idx == self.selected_shot_idx:
                    self.tree_shots.selection_set(item_id)
                    self.tree_shots.see(item_id)
        finally:
            self._suppress_tree_events = False

    def on_tree_select(self, event=None):
        if getattr(self, "_suppress_tree_events", False):
            return
        selected_items = self.tree_shots.selection()
        if selected_items:
            item_idx = self.tree_shots.index(selected_items[0])
            if self.selected_shot_idx == item_idx:
                return
            self.selected_shot_idx = item_idx
            self.redraw_canvas()
            if hasattr(self, "eagle_eye_dialog") and self.eagle_eye_dialog and self.eagle_eye_dialog.winfo_exists():
                self.eagle_eye_dialog.set_shot_index(item_idx)

    # =========================================================================
    # Interactive Manipulation: Add, Drag, Delete, Undo
    # =========================================================================

    def push_undo(self):
        state_copy = copy.deepcopy(self.shots)
        self.undo_stack.append(state_copy)
        if len(self.undo_stack) > 30:
            self.undo_stack.pop(0)
        self.is_modified = True
        self.lbl_save_status.config(text="● Unsaved changes", foreground="#ff9100")

    def on_undo(self):
        if self.undo_stack:
            self.shots = self.undo_stack.pop()
            self.selected_shot_idx = min(self.selected_shot_idx or 0, max(0, len(self.shots) - 1)) if self.shots else None
            self.recalculate_scores()
            self.redraw_canvas()
            self.lbl_save_status.config(text="↺ Undone", foreground="#00e5ff")

    def on_canvas_left_click(self, event):
        img_x, img_y = self.screen_to_img(event.x, event.y)
        self.drag_start_screen = (event.x, event.y)

        # 1. Bullseye Mode check
        if self.edit_mode_var.get() == "bullseye":
            self.push_undo()
            self.bullseye = (img_x, img_y)
            self.dragging_mode = "bullseye"
            self.recalculate_scores()
            self.redraw_canvas()
            return

        # 2. Check if clicked near Bullseye in Shot Mode
        dist_to_bull = math.hypot(img_x - self.bullseye[0], img_y - self.bullseye[1])
        if dist_to_bull <= (14.0 / self.zoom):
            self.push_undo()
            self.dragging_mode = "bullseye"
            self.drag_start_item_pos = self.bullseye
            return

        # 3. Hit-test existing shot markers
        clicked_shot_idx = None
        for idx, shot in enumerate(self.shots):
            dist_px = math.hypot(img_x - shot["x"], img_y - shot["y"])
            hit_radius = max(6.0 / self.zoom, (shot["caliber_mm"] / 2.0) * self.pixels_per_mm)
            if dist_px <= hit_radius + 4.0:
                clicked_shot_idx = idx
                break

        if clicked_shot_idx is not None:
            # Select and start dragging existing shot
            self.push_undo()
            self.selected_shot_idx = clicked_shot_idx
            self.dragging_mode = "shot"
            self.drag_start_item_pos = (self.shots[clicked_shot_idx]["x"], self.shots[clicked_shot_idx]["y"])
            self.redraw_canvas()
            self._select_tree_row(clicked_shot_idx)
            if hasattr(self, "eagle_eye_dialog") and self.eagle_eye_dialog and self.eagle_eye_dialog.winfo_exists():
                self.eagle_eye_dialog.set_shot_index(clicked_shot_idx)
        else:
            # Empty canvas click: Add new shot marker!
            if 0 <= img_x <= self.canvas_size and 0 <= img_y <= self.canvas_size:
                self.push_undo()
                new_id = (max([s["id"] for s in self.shots], default=0)) + 1
                new_shot = {
                    "id": new_id,
                    "x": float(img_x),
                    "y": float(img_y),
                    "score": 0,
                    "is_x": False,
                    "caliber_mm": self.caliber_mm,
                    "confidence": 1.0,
                }
                self.shots.append(new_shot)
                self.selected_shot_idx = len(self.shots) - 1
                self.recalculate_scores()
                self.redraw_canvas()
                self._select_tree_row(self.selected_shot_idx)
                if hasattr(self, "eagle_eye_dialog") and self.eagle_eye_dialog and self.eagle_eye_dialog.winfo_exists():
                    self.eagle_eye_dialog.set_shot_index(self.selected_shot_idx)
                self.lbl_status_msg.config(text=f"Added shot #{new_id} at ({img_x:.1f}, {img_y:.1f})")

    def on_canvas_double_click(self, event):
        """Double clicking on a shot opens the Eagle Eye scoring gauge."""
        img_x, img_y = self.screen_to_img(event.x, event.y)
        for idx, shot in enumerate(self.shots):
            dist_px = math.hypot(img_x - shot["x"], img_y - shot["y"])
            hit_radius = max(6.0 / self.zoom, (shot["caliber_mm"] / 2.0) * self.pixels_per_mm)
            if dist_px <= hit_radius + 4.0:
                self.selected_shot_idx = idx
                self._select_tree_row(idx)
                self.redraw_canvas()
                self.on_open_eagle_eye(toggle=False)
                return

    def close_eagle_eye(self):
        """Safely closes the Eagle Eye loupe dialog if open and refocuses parent."""
        if hasattr(self, "eagle_eye_dialog") and self.eagle_eye_dialog:
            dialog = self.eagle_eye_dialog
            self.eagle_eye_dialog = None
            try:
                if dialog.winfo_exists():
                    dialog.destroy()
            except Exception:
                pass
        try:
            self.focus_force()
        except Exception:
            pass

    def on_open_eagle_eye(self, toggle: bool = True):
        """Opens, toggles, or refocuses the magnified Eagle Eye Scoring Gauge dialog."""
        if not self.shots:
            messagebox.showinfo("Eagle Eye Gauge", "No shots to inspect. Click the canvas to place a shot first.")
            return

        if hasattr(self, "eagle_eye_dialog") and self.eagle_eye_dialog and self.eagle_eye_dialog.winfo_exists():
            if toggle:
                self.close_eagle_eye()
                return
            else:
                self.eagle_eye_dialog.set_shot_index(self.selected_shot_idx if self.selected_shot_idx is not None else 0)
                self.eagle_eye_dialog.lift()
                self.eagle_eye_dialog.focus_force()
                return

        if self.selected_shot_idx is None or not (0 <= self.selected_shot_idx < len(self.shots)):
            self.selected_shot_idx = 0
            self._select_tree_row(0)
            self.redraw_canvas()

        self.eagle_eye_dialog = EagleEyeLoupe(self, self.selected_shot_idx)

    def on_canvas_left_drag(self, event):
        img_x, img_y = self.screen_to_img(event.x, event.y)

        if self.dragging_mode == "shot" and self.selected_shot_idx is not None:
            # Drag selected shot
            clamped_x = max(0.0, min(float(self.canvas_size), img_x))
            clamped_y = max(0.0, min(float(self.canvas_size), img_y))
            self.shots[self.selected_shot_idx]["x"] = clamped_x
            self.shots[self.selected_shot_idx]["y"] = clamped_y
            self.recalculate_scores()
            self.redraw_canvas()
        elif self.dragging_mode == "bullseye":
            # Drag bullseye center
            self.bullseye = (max(0.0, min(float(self.canvas_size), img_x)), max(0.0, min(float(self.canvas_size), img_y)))
            self.recalculate_scores()
            self.redraw_canvas()

    def on_canvas_left_release(self, event):
        self.dragging_mode = None

    def on_canvas_right_click(self, event):
        """Right-click on a shot deletes it."""
        img_x, img_y = self.screen_to_img(event.x, event.y)
        for idx, shot in enumerate(self.shots):
            dist_px = math.hypot(img_x - shot["x"], img_y - shot["y"])
            hit_radius = max(6.0 / self.zoom, (shot["caliber_mm"] / 2.0) * self.pixels_per_mm)
            if dist_px <= hit_radius + 4.0:
                self.push_undo()
                del_id = shot["id"]
                del self.shots[idx]
                if not self.shots:
                    self.selected_shot_idx = None
                    self.close_eagle_eye()
                else:
                    self.selected_shot_idx = min(idx, len(self.shots) - 1)
                    self._select_tree_row(self.selected_shot_idx)
                    if hasattr(self, "eagle_eye_dialog") and self.eagle_eye_dialog and self.eagle_eye_dialog.winfo_exists():
                        self.eagle_eye_dialog.set_shot_index(self.selected_shot_idx)
                self.recalculate_scores()
                self.redraw_canvas()
                self.lbl_status_msg.config(text=f"Deleted shot #{del_id}")
                return

    def on_canvas_middle_click(self, event):
        self.drag_start_screen = (event.x, event.y)
        self.dragging_mode = "pan"

    def on_canvas_middle_drag(self, event):
        if self.dragging_mode == "pan":
            dx = event.x - self.drag_start_screen[0]
            dy = event.y - self.drag_start_screen[1]
            self.drag_start_screen = (event.x, event.y)
            self.pan_x -= dx / self.zoom
            self.pan_y -= dy / self.zoom
            self.redraw_canvas()

    def on_canvas_mouse_move(self, event):
        img_x, img_y = self.screen_to_img(event.x, event.y)
        dist_mm = math.hypot(img_x - self.bullseye[0], img_y - self.bullseye[1]) / self.pixels_per_mm
        self.lbl_cursor_pos.config(text=f"Cursor: X={img_x:.1f}, Y={img_y:.1f} (r={dist_mm:.1f} mm)")

    def nudge_selected_shot(self, dx: int, dy: int):
        if self.selected_shot_idx is not None and 0 <= self.selected_shot_idx < len(self.shots):
            self.push_undo()
            self.shots[self.selected_shot_idx]["x"] = max(0.0, min(float(self.canvas_size), self.shots[self.selected_shot_idx]["x"] + dx))
            self.shots[self.selected_shot_idx]["y"] = max(0.0, min(float(self.canvas_size), self.shots[self.selected_shot_idx]["y"] + dy))
            self.recalculate_scores()
            self.redraw_canvas()

    def on_delete_selected(self):
        if self.selected_shot_idx is not None and 0 <= self.selected_shot_idx < len(self.shots):
            self.push_undo()
            del_id = self.shots[self.selected_shot_idx]["id"]
            del self.shots[self.selected_shot_idx]
            if not self.shots:
                self.selected_shot_idx = None
                self.close_eagle_eye()
            else:
                self.selected_shot_idx = min(self.selected_shot_idx, len(self.shots) - 1)
                self._select_tree_row(self.selected_shot_idx)
                if hasattr(self, "eagle_eye_dialog") and self.eagle_eye_dialog and self.eagle_eye_dialog.winfo_exists():
                    self.eagle_eye_dialog.set_shot_index(self.selected_shot_idx)
            self.recalculate_scores()
            self.redraw_canvas()
            self.lbl_status_msg.config(text=f"Deleted shot #{del_id}")

    def on_clear_shots(self):
        if not self.shots:
            messagebox.showinfo("Delete All Shots", "There are no shots on this target.")
            return
        shot_count = len(self.shots)
        if messagebox.askyesno("Delete All Shots", f"Are you sure you want to remove all {shot_count} shot markers on this target?"):
            self.push_undo()
            self.shots.clear()
            self.selected_shot_idx = None
            self.close_eagle_eye()
            self.recalculate_scores()
            self.redraw_canvas()
            self.lbl_status_msg.config(text=f"Removed all {shot_count} shots. Click canvas to add correct bullet holes.")

    def on_renumber_shots(self):
        self.push_undo()
        for i, s in enumerate(self.shots, 1):
            s["id"] = i
        self.recalculate_scores()
        self.redraw_canvas()

    def on_revert(self):
        if self.current_idx >= 0:
            self.load_target_index(self.current_idx)

    def _select_tree_row(self, row_idx: int):
        children = self.tree_shots.get_children()
        if 0 <= row_idx < len(children):
            self._suppress_tree_events = True
            try:
                self.tree_shots.selection_set(children[row_idx])
                self.tree_shots.see(children[row_idx])
            finally:
                self._suppress_tree_events = False

    def on_browse_copy_dir(self):
        current_dir = self.copy_dir_var.get().strip() or os.getcwd()
        selected = filedialog.askdirectory(
            title="Select Destination Directory to Copy Saved Targets",
            initialdir=current_dir,
            parent=self
        )
        if selected:
            self.copy_dir_var.set(selected)
            self.copy_on_save_var.set(True)

    def _copy_target_to_directory(
        self,
        dest_dir: Path,
        updated_ann: TargetAnnotation,
        json_p: Path,
        review_p: Path
    ) -> List[str]:
        """
        Copies raw photo, canonical image, annotation JSON, and review card to destination directory.
        Respects existing structured subfolders (annotations/, canonical_1024/, raw/) if present.
        Returns list of copied asset names for status reporting.
        """
        dest_dir.mkdir(parents=True, exist_ok=True)
        copied_assets: List[str] = []

        # Determine if destination uses structured subfolders
        has_subfolders = (dest_dir / "annotations").is_dir() or (dest_dir / "canonical_1024").is_dir()

        if has_subfolders:
            ann_dest = dest_dir / "annotations" / json_p.name
            canon_dest = dest_dir / "canonical_1024" / f"{updated_ann.target_id}_canonical.png"
            review_dest = dest_dir / "canonical_1024" / f"{updated_ann.target_id}_review.png"
            raw_dest_dir = dest_dir / "raw"
        else:
            ann_dest = dest_dir / json_p.name
            canon_dest = dest_dir / f"{updated_ann.target_id}_canonical.png"
            review_dest = dest_dir / f"{updated_ann.target_id}_review.png"
            raw_dest_dir = dest_dir / "raw" if (dest_dir / "raw").is_dir() else dest_dir

        ann_dest.parent.mkdir(parents=True, exist_ok=True)
        canon_dest.parent.mkdir(parents=True, exist_ok=True)
        review_dest.parent.mkdir(parents=True, exist_ok=True)
        raw_dest_dir.mkdir(parents=True, exist_ok=True)

        # 1. Copy / save updated annotation JSON
        if json_p.resolve() != ann_dest.resolve():
            updated_ann.to_json(ann_dest)
        copied_assets.append("JSON")

        # 2. Copy canonical image
        canon_src = self._resolve_canonical_image_path(json_p, updated_ann)
        if canon_src and canon_src.exists():
            if canon_src.resolve() != canon_dest.resolve():
                shutil.copy2(canon_src, canon_dest)
            copied_assets.append("Canonical")
        elif self.canonical_cv2 is not None:
            cv2.imwrite(str(canon_dest), self.canonical_cv2)
            copied_assets.append("Canonical")

        # 3. Copy raw photo (if found)
        raw_src = self._resolve_raw_image_path(json_p, updated_ann)
        if raw_src and raw_src.exists():
            dest_raw_path = raw_dest_dir / raw_src.name
            if raw_src.resolve() != dest_raw_path.resolve():
                shutil.copy2(raw_src, dest_raw_path)
            copied_assets.append(f"Raw ({raw_src.name})")

        # 4. Copy review card (if generated)
        if review_p.exists():
            if review_p.resolve() != review_dest.resolve():
                shutil.copy2(review_p, review_dest)
            copied_assets.append("Review")
        elif self.canonical_cv2 is not None:
            render_review_card(
                canonical_bgr=self.canonical_cv2,
                annotation=updated_ann,
                output_path=review_dest,
                redact_pii=updated_ann.metadata.get("redact_pii", False)
            )
            copied_assets.append("Review")

        return copied_assets

    # =========================================================================
    # Saving to JSON and Updating Review Card
    # =========================================================================

    def on_save(self):
        if not self.current_json_path or not self.current_ann:
            return

        # Build AnnotatedShot list
        new_shots: List[AnnotatedShot] = []
        for s in self.shots:
            new_shots.append(AnnotatedShot(
                id=int(s["id"]),
                center_x=float(s["x"]),
                center_y=float(s["y"]),
                score=int(s.get("score", 0)),
                is_x=bool(s.get("is_x", False)),
                caliber_mm=float(s.get("caliber_mm", self.caliber_mm)),
                shot_type="single",
                confidence=1.0,
            ))

        # Update TargetAnnotation
        updated_ann = TargetAnnotation(
            target_id=self.current_ann.target_id,
            image_filename=self.current_ann.image_filename,
            target_type=self.target_type,
            bullseye_center=self.bullseye,
            shots=new_shots,
            canvas_size=self.canvas_size,
            pixels_per_mm=self.pixels_per_mm,
            version=self.current_ann.version,
            metadata=dict(self.current_ann.metadata)
        )

        m = self.group_metrics
        if m:
            updated_ann.metadata["total_score"] = m.total_score
            updated_ann.metadata["x_count"] = m.x_count
            updated_ann.metadata["extreme_spread_mm"] = round(m.extreme_spread_mm, 2)
            updated_ann.metadata["mean_radius_mm"] = round(m.mean_radius_mm, 2)
        updated_ann.metadata["manually_verified"] = True
        updated_ann.metadata["verified_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        # 1. Save JSON to disk
        updated_ann.to_json(self.current_json_path)

        # 2. Regenerate Review Card image on disk
        review_path = self.current_json_path.parent / f"{updated_ann.target_id}_review.png"
        if self.canonical_cv2 is not None:
            render_review_card(
                canonical_bgr=self.canonical_cv2,
                annotation=updated_ann,
                output_path=review_path,
                redact_pii=self.current_ann.metadata.get("redact_pii", False)
            )

        # 3. Optional copy to destination directory
        copy_msg = ""
        if self.copy_on_save_var.get():
            dest_str = self.copy_dir_var.get().strip()
            if dest_str:
                dest_dir = Path(dest_str)
                try:
                    copied_items = self._copy_target_to_directory(
                        dest_dir=dest_dir,
                        updated_ann=updated_ann,
                        json_p=self.current_json_path,
                        review_p=review_path
                    )
                    copy_msg = f" | Copied {', '.join(copied_items)} -> '{dest_dir.name}'"
                except Exception as e:
                    copy_msg = f" | [!] Copy failed: {e}"

        self.is_modified = False
        self.lbl_save_status.config(text=f"✓ Saved ({len(new_shots)} shots)", foreground="#00ff78")
        self.lbl_status_msg.config(text=f"Successfully saved '{self.current_json_path.name}' and updated review card.{copy_msg}")


def launch_annotation_editor(
    target_path: Optional[Union[str, Path]] = None,
    target_type: Optional[str] = None,
    copy_to_dir: Optional[Union[str, Path]] = None
):
    """Entry point for running the Annotation Editor GUI."""
    try:
        app = AnnotationEditor(
            target_dir_or_file=target_path,
            target_type_override=target_type,
            copy_to_dir=copy_to_dir
        )
        app.mainloop()
    except tk.TclError as e:
        if "display" in str(e).lower():
            print("[-] Error: Could not connect to graphical display server ($DISPLAY or Wayland).")
            print("    The Interactive Annotation Editor requires an X11 or Wayland desktop environment.")
            print("    If connecting remotely via SSH, use 'ssh -X' or 'ssh -Y' to enable X11 forwarding.")
            sys.exit(1)
        raise


if __name__ == "__main__":
    path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    launch_annotation_editor(path_arg)
