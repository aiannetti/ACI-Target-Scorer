"""
Procedural target heatmap training pipeline.
"""

import argparse
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from target_scorer.config import TrainConfig
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
    """Evaluates sub-millimeter Mean Radial Error (MRE) between detected peaks and ground truth."""
    detected_peaks = extract_subpixel_peaks(pred_hmap, threshold=threshold)
    if not detected_peaks or len(gt_coords) == 0:
        return 999.0, 999.0, 0, len(gt_coords)

    det_points = np.array([[p[0], p[1]] for p in detected_peaks], dtype=np.float32)

    diff = gt_coords[:, np.newaxis, :] - det_points[np.newaxis, :, :]
    dist_px = np.sqrt(np.sum(diff ** 2, axis=-1))

    min_dists_px = np.min(dist_px, axis=1)
    match_threshold_px = 8.0 * pixels_per_mm
    valid_matches = min_dists_px[min_dists_px < match_threshold_px]

    if len(valid_matches) == 0:
        return 999.0, 999.0, 0, len(gt_coords)

    errors_mm = valid_matches / pixels_per_mm
    mean_err_mm = float(np.mean(errors_mm))
    max_err_mm = float(np.max(errors_mm))

    return mean_err_mm, max_err_mm, len(valid_matches), len(gt_coords)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    config: TrainConfig,
    scaler: torch.amp.GradScaler = None,
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
                if mre_mm < 900.0:
                    all_errors_mm.append(mre_mm)

    val_loss = total_loss / len(dataloader)
    mean_mre = float(np.mean(all_errors_mm)) if all_errors_mm else 999.0
    total_samples = len(dataloader) * config.batch_size
    match_rate = (len(all_errors_mm) / total_samples * 100.0) if total_samples > 0 else 0.0
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
    use_amp: bool = True,
) -> str:
    """Initializes and runs the full end-to-end training and evaluation pipeline."""
    config = TrainConfig(
        canvas_size=canvas_size,
        pixels_per_mm=pixels_per_mm,
        train_samples_per_epoch=train_samples_per_epoch,
        val_samples_per_epoch=val_samples_per_epoch,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        peak_threshold=0.45,
        num_workers=2,
        pin_memory=True,
        use_amp=use_amp,
    )

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        device_name = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    else:
        device_name = "CPU"
        total_vram_gb = 0.0

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=config.use_amp and (config.device == "cuda") and torch.cuda.is_available()
    )

    print("=" * 72)
    print("  HIGH-RESOLUTION HEATMAP REGRESSION FOR TARGET SCORING")
    print(f"  Target Device:       {config.device.upper()} ({device_name})")
    print(f"  Available VRAM:      {total_vram_gb:.2f} GB")
    print(f"  Canvas Dimensions:   {config.canvas_size} x {config.canvas_size} px")
    print(f"  Metrology Scale:     {config.pixels_per_mm} pixels/mm")
    print(f"  Mixed Precision:     {'Enabled (AMP)' if scaler.is_enabled() else 'Disabled (FP32)'}")
    print(f"  Worker Threads:      {config.num_workers} (Background CPU rendering)")
    print(f"  A-Wing Parameters:   omega={config.awing_omega}, alpha={config.awing_alpha}, theta={config.awing_theta}")
    print("=" * 72)

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

    criterion = AdaptiveWingLoss(
        omega=config.awing_omega,
        theta=config.awing_theta,
        epsilon=config.awing_epsilon,
        alpha=config.awing_alpha
    )
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)

    best_mre = float("inf")

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
            f"Match Rate: {match_rate:.1f}%"
        )

        if val_mre_mm < best_mre and val_mre_mm < 900.0:
            best_mre = val_mre_mm
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_mre_mm": best_mre,
                "config": config,
            }, checkpoint_path)
            print(f"  --> Saved new best checkpoint to '{checkpoint_path}' (MRE: {best_mre:.3f} mm)")

    print("=" * 72)
    print(f"[✓] Training complete. Optimal checkpoint: {checkpoint_path}")
    print(f"[✓] Peak accuracy achieved: {best_mre:.3f} mm sub-pixel error.")
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
    parser.add_argument("--amp", action="store_true", default=True, help="Enable automatic mixed precision (AMP)")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable automatic mixed precision")

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
    )


if __name__ == "__main__":
    main()
