"""
Interactive GUI for manipulating shot markers, live target metrology, and JSON annotation correction.
Allows rapid point-and-click correction of missed or misidentified bullet holes on target review images.
"""

import copy
import json
import math
import os
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


class AnnotationEditor(tk.Tk):
    """
    High-performance interactive desktop GUI for reviewing, adding, moving, and deleting
    bullet hole markers on autolabeled target cards, with real-time NRA metrology re-scoring.
    """

    def __init__(self, target_dir_or_file: Optional[Union[str, Path]] = None, target_type_override: Optional[str] = None):
        super().__init__()

        self.title("Target Scorer - Interactive Shot Annotation & Metrology Editor")
        self.geometry("1420x920")
        self.minsize(1050, 700)

        # Apply dark theme styling
        self.configure(bg="#18191c")
        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self._configure_theme()

        # State
        self.target_type_override = target_type_override
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

        # Modes: 'shot' (add/move shots), 'bullseye' (move bullseye)
        self.edit_mode_var = tk.StringVar(value="shot")
        self.show_rings_var = tk.BooleanVar(value=True)
        self.show_spread_var = tk.BooleanVar(value=True)
        self.show_badges_var = tk.BooleanVar(value=True)

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

    def _configure_theme(self):
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

        self.style.configure("TCombobox", fieldbackground="#2b2d38", background="#323440", foreground=text_fg)
        self.style.configure("TCheckbutton", background=dark_bg, foreground=text_fg)
        self.style.configure("TRadiobutton", background=dark_bg, foreground=text_fg)

        self.style.configure("Treeview", background="#22232a", foreground=text_fg, fieldbackground="#22232a", rowheight=24)
        self.style.map("Treeview", background=[("selected", "#005a9e")], foreground=[("selected", "#ffffff")])
        self.style.configure("Treeview.Heading", background="#2c2d36", foreground=text_fg, font=("Segoe UI", 9, "bold"))

    def _build_ui(self):
        # 1. Top Control Toolbar
        top_bar = ttk.Frame(self, padding=(10, 8))
        top_bar.pack(side=tk.TOP, fill=tk.X)

        btn_open = ttk.Button(top_bar, text="📁 Open Directory...", command=self.on_open_directory)
        btn_open.pack(side=tk.LEFT, padx=(0, 10))

        btn_prev = ttk.Button(top_bar, text="◀ Prev (P)", command=self.on_prev_target)
        btn_prev.pack(side=tk.LEFT, padx=2)

        self.lbl_target_info = ttk.Label(top_bar, text="No targets loaded", font=("Segoe UI", 10, "bold"), width=36)
        self.lbl_target_info.pack(side=tk.LEFT, padx=6)

        btn_next = ttk.Button(top_bar, text="Next (N) ▶", command=self.on_next_target)
        btn_next.pack(side=tk.LEFT, padx=2)

        ttk.Separator(top_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=12)

        # Mode Selection
        ttk.Label(top_bar, text="Mode:").pack(side=tk.LEFT, padx=(0, 4))
        rb_shot = ttk.Radiobutton(top_bar, text="Shots (Click to Add / Drag)", variable=self.edit_mode_var, value="shot")
        rb_shot.pack(side=tk.LEFT, padx=4)
        rb_bull = ttk.Radiobutton(top_bar, text="Bullseye Center", variable=self.edit_mode_var, value="bullseye")
        rb_bull.pack(side=tk.LEFT, padx=4)

        ttk.Separator(top_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=12)

        # Caliber Selector
        ttk.Label(top_bar, text="Caliber:").pack(side=tk.LEFT, padx=(0, 4))
        self.caliber_combo = ttk.Combobox(
            top_bar,
            values=[".22 LR (5.69 mm)", "9mm / .38 (9.02 mm)", ".45 ACP (11.48 mm)"],
            state="readonly",
            width=18
        )
        self.caliber_combo.current(0)
        self.caliber_combo.bind("<<ComboboxSelected>>", self.on_caliber_changed)
        self.caliber_combo.pack(side=tk.LEFT, padx=4)

        # Right-aligned buttons on top toolbar
        btn_save = ttk.Button(top_bar, text="💾 Save JSON & Review (Ctrl+S)", style="Success.TButton", command=self.on_save)
        btn_save.pack(side=tk.RIGHT, padx=4)

        btn_reset = ttk.Button(top_bar, text="↺ Revert", command=self.on_revert)
        btn_reset.pack(side=tk.RIGHT, padx=4)

        btn_clear = ttk.Button(top_bar, text="✕ Clear Shots", style="Danger.TButton", command=self.on_clear_shots)
        btn_clear.pack(side=tk.RIGHT, padx=4)

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

        cb_rings = ttk.Checkbutton(canvas_tools, text="NRA Ring Guides", variable=self.show_rings_var, command=self.redraw_canvas)
        cb_rings.pack(side=tk.LEFT, padx=8)

        cb_spread = ttk.Checkbutton(canvas_tools, text="Extreme Spread Line", variable=self.show_spread_var, command=self.redraw_canvas)
        cb_spread.pack(side=tk.LEFT, padx=8)

        cb_badges = ttk.Checkbutton(canvas_tools, text="Shot Badges", variable=self.show_badges_var, command=self.redraw_canvas)
        cb_badges.pack(side=tk.LEFT, padx=8)

        # Tkinter Canvas
        self.canvas = tk.Canvas(canvas_frame, bg="#101114", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Canvas Event Bindings
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.canvas.bind("<Button-1>", self.on_canvas_left_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_left_release)

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

        # Bottom Action Buttons on Sidebar
        side_actions = ttk.Frame(hud_frame, padding=(0, 8))
        side_actions.pack(side=tk.BOTTOM, fill=tk.X)

        btn_del_shot = ttk.Button(side_actions, text="Delete Selected Shot (Del)", style="Danger.TButton", command=self.on_delete_selected)
        btn_del_shot.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        btn_undo = ttk.Button(side_actions, text="Undo (Ctrl+Z)", command=self.on_undo)
        btn_undo.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))

        # 3. Bottom Status Bar
        status_bar = ttk.Frame(self, relief="sunken", padding=(8, 4))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.lbl_cursor_pos = ttk.Label(status_bar, text="Cursor: X=0.0, Y=0.0", width=34)
        self.lbl_cursor_pos.pack(side=tk.LEFT)

        self.lbl_status_msg = ttk.Label(status_bar, text="Ready. Click canvas to add shots | Drag to move | Right-click to remove", foreground="#00d7ff")
        self.lbl_status_msg.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        self.lbl_save_status = ttk.Label(status_bar, text="", font=("Segoe UI", 9, "bold"), foreground="#00ff78")
        self.lbl_save_status.pack(side=tk.RIGHT)

    def _bind_shortcuts(self):
        self.bind("<Control-s>", lambda e: self.on_save())
        self.bind("<Control-z>", lambda e: self.on_undo())
        self.bind("<Delete>", lambda e: self.on_delete_selected())
        self.bind("<BackSpace>", lambda e: self.on_delete_selected())
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

    def _update_caliber_combobox(self):
        cal = self.caliber_mm
        if abs(cal - 5.69) < 0.5:
            self.caliber_combo.current(0)
        elif abs(cal - 9.02) < 0.5:
            self.caliber_combo.current(1)
        elif abs(cal - 11.48) < 0.5:
            self.caliber_combo.current(2)

    def on_caliber_changed(self, event=None):
        idx = self.caliber_combo.current()
        if idx == 0:
            self.caliber_mm = 5.69
        elif idx == 1:
            self.caliber_mm = 9.02
        elif idx == 2:
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

    def on_tree_select(self, event):
        selected_items = self.tree_shots.selection()
        if selected_items:
            item_idx = self.tree_shots.index(selected_items[0])
            self.selected_shot_idx = item_idx
            self.redraw_canvas()

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
                self.lbl_status_msg.config(text=f"Added shot #{new_id} at ({img_x:.1f}, {img_y:.1f})")

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
                self.selected_shot_idx = None
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
            self.selected_shot_idx = None
            self.recalculate_scores()
            self.redraw_canvas()
            self.lbl_status_msg.config(text=f"Deleted shot #{del_id}")

    def on_clear_shots(self):
        if not self.shots:
            return
        if messagebox.askyesno("Clear All Shots", "Are you sure you want to remove all shot markers on this target?"):
            self.push_undo()
            self.shots.clear()
            self.selected_shot_idx = None
            self.recalculate_scores()
            self.redraw_canvas()
            self.lbl_status_msg.config(text="Cleared all shot markers. Click canvas to add correct bullet holes.")

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
            self.tree_shots.selection_set(children[row_idx])
            self.tree_shots.see(children[row_idx])

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

        self.is_modified = False
        self.lbl_save_status.config(text=f"✓ Saved ({len(new_shots)} shots)", foreground="#00ff78")
        self.lbl_status_msg.config(text=f"Successfully saved '{self.current_json_path.name}' and updated review card.")


def launch_annotation_editor(target_path: Optional[Union[str, Path]] = None, target_type: Optional[str] = None):
    """Entry point for running the Annotation Editor GUI."""
    app = AnnotationEditor(target_dir_or_file=target_path, target_type_override=target_type)
    app.mainloop()


if __name__ == "__main__":
    path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    launch_annotation_editor(path_arg)
