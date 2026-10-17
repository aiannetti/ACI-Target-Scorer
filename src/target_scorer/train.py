"""
Procedural target heatmap training pipeline.
"""

import argparse
import os
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from target_scorer.config import TrainConfig
from target_scorer.data.hybrid_dataset import HybridTargetDataset
from target_scorer.metrology.peaks import extract_subpixel_peaks
from target_scorer.models.loss import AdaptiveWingLoss
from target_scorer.models.unet import HighResHeatmapUNet
from target_scorer.synthetic.dataset import SyntheticTargetDataset, custom_collate_fn


def calculate_localization_metrics(
    pred_hmap: np.ndarray, 
    gt_coords: np.ndarray, 
    pixels_per_mm: float, 
    threshold: float = 0.35
) -> Tuple[float, float, int, int]:
    """Evaluates sub-millimeter Mean Radial Error (MRE) via 1-to-1 greedy bipartite matching."""
    min_dist_px = max(6.0, 2.5 * pixels_per_mm)
    detected_peaks = extract_subpixel_peaks(
        pred_hmap, threshold=threshold, min_distance_px=min_dist_px, max_peaks=20
    )
    num_gt = len(gt_coords)
    if not detected_peaks or num_gt == 0:
        return 999.0, 999.0, 0, num_gt

    det_points = np.array([[p[0], p[1]] for p in detected_peaks], dtype=np.float32)

    diff = gt_coords[:, np.newaxis, :] - det_points[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

    match_threshold_px = 6.0 * pixels_per_mm
    matched_dists_px: List[float] = []
    used_gt = set()
    used_det = set()

    # Sort all (gt_idx, det_idx) pairs by Euclidean distance for 1-to-1 greedy assignment
    pairs = [
        (float(dist_matrix[g_i, d_i]), g_i, d_i)
        for g_i in range(num_gt)
        for d_i in range(len(det_points))
        if dist_matrix[g_i, d_i] < match_threshold_px
    ]
    pairs.sort(key=lambda x: x[0])

    for d_px, g_i, d_i in pairs:
        if g_i not in used_gt and d_i not in used_det:
            used_gt.add(g_i)
            used_det.add(d_i)
            matched_dists_px.append(d_px)

    if not matched_dists_px:
        return 999.0, 999.0, 0, num_gt

    errors_mm = np.array(matched_dists_px, dtype=np.float32) / pixels_per_mm
    mean_err_mm = float(np.mean(errors_mm))
    max_err_mm = float(np.max(errors_mm))

    return mean_err_mm, max_err_mm, len(matched_dists_px), num_gt


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    config: TrainConfig,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> float:
    """Executes one training epoch with gradient clipping and loss accumulation."""
    model.train()
    total_loss = 0.0
    use_scaler = scaler is not None and scaler.is_enabled()

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(config.device)
        targets = batch["heatmap"].to(config.device)

        optimizer.zero_grad()
        if use_scaler:
            with torch.amp.autocast("cuda", enabled=True):
                predictions = model(images)
                loss = criterion(predictions, targets)
            scaler.scale(loss).backward()
            if config.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            predictions = model(images)
            loss = criterion(predictions, targets)
            loss.backward()
            if config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    config: TrainConfig,
) -> Tuple[float, float, float]:
    """Validates loss and computes sub-pixel Mean Radial Error (MRE) in millimeters."""
    model.eval()
    total_loss = 0.0
    all_errors_mm: List[float] = []
    total_matched_shots = 0
    total_gt_shots = 0
    use_amp = config.use_amp and config.device == "cuda" and torch.cuda.is_available()

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(config.device)
            targets = batch["heatmap"].to(config.device)

            if use_amp:
                with torch.amp.autocast("cuda", enabled=True):
                    predictions = model(images)
                    loss = criterion(predictions, targets)
            else:
                predictions = model(images)
                loss = criterion(predictions, targets)
            total_loss += loss.item()

            preds_np = predictions.detach().float().cpu().numpy()
            gt_coords_batch = batch["shot_coords"]

            for b in range(images.size(0)):
                pred_shot_hmap = preds_np[b, 0]
                sample_coords = gt_coords_batch[b]
                gt_coords = (
                    sample_coords.cpu().numpy()
                    if isinstance(sample_coords, torch.Tensor)
                    else np.array(sample_coords, dtype=np.float32)
                )
                mre_mm, _, matches, total = calculate_localization_metrics(
                    pred_shot_hmap, gt_coords, config.pixels_per_mm, threshold=config.peak_threshold
                )
                total_matched_shots += matches
                total_gt_shots += total
                if mre_mm < 900.0:
                    all_errors_mm.append(mre_mm)

    val_loss = total_loss / len(dataloader)
    mean_mre = float(np.mean(all_errors_mm)) if all_errors_mm else 999.0
    match_rate = (total_matched_shots / total_gt_shots * 100.0) if total_gt_shots > 0 else 0.0
    return val_loss, mean_mre, match_rate


def run_training(
    canvas_size: int = 512,
    pixels_per_mm: float = 4.0,
    epochs: int = 15,
    batch_size: int = 4,
    learning_rate: float = 3.5e-4,
    train_samples_per_epoch: int = 400,
    val_samples_per_epoch: int = 50,
    checkpoint_path: str = "target_scoring_hrnet_best.pt",
    use_amp: Optional[bool] = None,
    num_workers: int = 2,
    sigma: Optional[float] = None,
    initial_weights: Optional[str] = None,
    real_data_dir: Optional[str] = None,
    real_ratio: float = 0.35,
) -> str:
    """Initializes and runs the full end-to-end training and evaluation pipeline."""
    if canvas_size % 8 != 0:
        raise ValueError(
            f"Canvas size ({canvas_size}) must be divisible by 8 for U-Net architecture."
        )
    has_tensor_cores = False
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        device_name = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        cc_major, cc_minor = torch.cuda.get_device_capability(0)
        has_tensor_cores = cc_major >= 7
    else:
        device_name = "CPU"
        total_vram_gb = 0.0
        cc_major, cc_minor = (0, 0)

    # Auto-detect AMP support: enable only on Volta+ (CC >= 7.0) with Tensor Cores unless overridden
    if use_amp is None:
        effective_amp = has_tensor_cores
    else:
        effective_amp = use_amp and has_tensor_cores

    if sigma is not None:
        scaled_sigma = float(sigma)
    else:
        scaled_sigma = max(1.8, 2.5 * (pixels_per_mm / 4.0))

    config = TrainConfig(
        canvas_size=canvas_size,
        pixels_per_mm=pixels_per_mm,
        heatmap_sigma_px=scaled_sigma,
        train_samples_per_epoch=train_samples_per_epoch,
        val_samples_per_epoch=val_samples_per_epoch,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        peak_threshold=0.40,
        num_workers=num_workers,
        pin_memory=True,
        use_amp=effective_amp,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=config.use_amp and (config.device == "cuda") and torch.cuda.is_available()
    )

    if scaler.is_enabled():
        amp_status = f"Enabled (AMP - CC {cc_major}.{cc_minor} Tensor Cores)"
    elif torch.cuda.is_available() and not has_tensor_cores:
        amp_status = f"Disabled (Native FP32 - CC {cc_major}.{cc_minor} Pascal lacks Tensor Cores)"
    else:
        amp_status = "Disabled (FP32)"

    print("=" * 72)
    print("  HIGH-RESOLUTION HEATMAP REGRESSION FOR TARGET SCORING")
    print(f"  Target Device:       {config.device.upper()} ({device_name})")
    print(f"  Available VRAM:      {total_vram_gb:.2f} GB")
    print(f"  Canvas Dimensions:   {config.canvas_size} x {config.canvas_size} px")
    print(f"  Metrology Scale:     {config.pixels_per_mm} pixels/mm")
    print(f"  Mixed Precision:     {amp_status}")
    print(f"  Worker Threads:      {config.num_workers} (Background CPU rendering)")
    if real_data_dir:
        print(f"  Real Data Directory: {real_data_dir} (Ratio: {real_ratio:.2f})")
    print(f"  A-Wing Parameters:   omega={config.awing_omega}, alpha={config.awing_alpha}, theta={config.awing_theta}")
    print("=" * 72)

    if real_data_dir is not None:
        print(f"[*] Ingesting real target dataset from '{real_data_dir}' (Real Ratio: {real_ratio:.2f})")
        train_dataset = HybridTargetDataset(
            real_data_dir=real_data_dir,
            config=config,
            size=config.train_samples_per_epoch,
            real_ratio=real_ratio,
            is_train=True,
        )
        val_dataset = HybridTargetDataset(
            real_data_dir=real_data_dir,
            config=config,
            size=config.val_samples_per_epoch,
            real_ratio=real_ratio,
            is_train=False,
        )
    else:
        train_dataset = SyntheticTargetDataset(size=config.train_samples_per_epoch, config=config, is_train=True)
        val_dataset = SyntheticTargetDataset(size=config.val_samples_per_epoch, config=config, is_train=False)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and (config.device == "cuda"),
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and (config.device == "cuda"),
        collate_fn=custom_collate_fn
    )

    model = HighResHeatmapUNet(in_channels=3, out_channels=2, base_dim=32).to(config.device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Initialized HighResHeatmapUNet (Total Trainable Parameters: {total_params:,})")

    if initial_weights is not None:
        if os.path.exists(initial_weights):
            print(f"[+] Loading initial weights for fine-tuning from '{initial_weights}'...")
            try:
                ckpt = torch.load(initial_weights, map_location=config.device, weights_only=False)
            except TypeError:
                ckpt = torch.load(initial_weights, map_location=config.device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict)
            print(f"[✓] Initial weights loaded successfully. Warm-starting from pre-trained state.")
        else:
            print(f"[-] Warning: Specified initial weights path '{initial_weights}' not found. Initializing from scratch.")

    criterion = AdaptiveWingLoss(
        omega=config.awing_omega,
        theta=config.awing_theta,
        epsilon=config.awing_epsilon,
        alpha=config.awing_alpha
    )
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)

    best_mre = float("inf")
    best_match_rate = 0.0
    best_composite_score = -float("inf")
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, config.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, config, scaler=scaler)
        val_loss, val_mre_mm, match_rate = validate_one_epoch(model, val_loader, criterion, config)
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch:02d}/{config.epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss (A-Wing): {train_loss:.5f} | "
            f"Val Loss: {val_loss:.5f} | "
            f"Val Sub-pixel MRE: {val_mre_mm:.3f} mm | "
            f"Match Rate: {match_rate:.1f}%",
            flush=True
        )

        # Composite metric balancing recall (match rate) and sub-pixel localization (MRE):
        # In competition scoring, missing an impact is far worse than a 0.02 mm localization difference.
        # Score = Match Rate (%) - 10.0 * MRE (mm)
        composite_score = match_rate - (10.0 * val_mre_mm if val_mre_mm < 900.0 else 100.0)

        if composite_score > best_composite_score and match_rate > 50.0:
            best_composite_score = composite_score
            best_mre = val_mre_mm
            best_match_rate = match_rate
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_mre_mm": best_mre,
                "match_rate": best_match_rate,
                "composite_score": best_composite_score,
                "config": config,
            }, checkpoint_path)
            print(f"  --> Saved new best checkpoint to '{checkpoint_path}' (Match: {best_match_rate:.1f}%, MRE: {best_mre:.3f} mm)", flush=True)

    print("=" * 72, flush=True)
    print(f"[✓] Training complete. Optimal checkpoint: {checkpoint_path}", flush=True)
    print(f"[✓] Peak accuracy achieved: {best_mre:.3f} mm sub-pixel error.", flush=True)
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(description="Procedural Target Heatmap Training Pipeline (NRA B-2 & B-3)")
    parser.add_argument("--canvas-size", type=int, default=512, help="Canonical canvas resolution (default: 512, use 256 for fast verification)")
    parser.add_argument("--scale", type=float, default=4.0, help="Pixels per mm metric scale (default: 4.0, use 2.0 for 256x256)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--lr", type=float, default=3.5e-4, help="Initial learning rate (default: 3.5e-4)")
    parser.add_argument("--train-samples", type=int, default=400, help="Procedural training samples per epoch (default: 400)")
    parser.add_argument("--val-samples", type=int, default=50, help="Validation samples per epoch (default: 50)")
    parser.add_argument("--checkpoint", type=str, default="target_scoring_hrnet_best.pt", help="Path to save best model checkpoint")
    parser.add_argument("--workers", type=int, default=2, help="Number of background CPU worker processes for data loading (default: 2)")
    parser.add_argument("--sigma", type=float, default=None, help="Gaussian heatmap sigma in pixels (default: scaled from px/mm)")
    parser.add_argument("--amp", action="store_true", default=True, help="Enable automatic mixed precision (AMP)")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable automatic mixed precision")
    parser.add_argument("--initial-weights", type=str, default=None, help="Path to existing weights checkpoint to warm-start/fine-tune from")
    parser.add_argument("--real-data-dir", type=str, default=None, help="Path to real target dataset directory or manifest (enables real/hybrid training)")
    parser.add_argument("--real-ratio", type=float, default=0.35, help="Sampling probability for real target images (1.0 = pure real fine-tuning, 0.35 = hybrid)")

    args = parser.parse_args()

    run_training(
        canvas_size=args.canvas_size,
        pixels_per_mm=args.scale,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        train_samples_per_epoch=args.train_samples,
        val_samples_per_epoch=args.val_samples,
        checkpoint_path=args.checkpoint,
        use_amp=args.amp,
        num_workers=args.workers,
        sigma=args.sigma,
        initial_weights=args.initial_weights,
        real_data_dir=args.real_data_dir,
        real_ratio=args.real_ratio,
    )


if __name__ == "__main__":
    main()
