"""Phase 5 — 5-fold cross-validation within ACDC only.

Per protocol 5.2/5.3: fix architecture/hyperparameters/preprocessing based
on ACDC CV only; beat a naive baseline; check for overfitting; do NOT touch
M&Ms. M&Ms is not referenced anywhere in this script.
"""

import configparser
import csv
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase5_model import SmallMotionCNN  # noqa: E402
from registration import dice_per_label, warp  # noqa: E402
from seed import set_all_seeds  # noqa: E402

DOPPLER_DIR = Path("data/processed/ACDC_doppler")
PROC_DIR = Path("data/processed/ACDC")
ACDC_ROOT = Path("data/ACDC/ACDC/database")
N_FOLDS = 5
N_EPOCHS = 30
LR = 1e-3
SEED = 42


def get_group(pid: str, split: str) -> str:
    cfg_path = ACDC_ROOT / split / pid / "Info.cfg"
    cfg_text = "[info]\n" + cfg_path.read_text()
    parser = configparser.ConfigParser()
    parser.read_string(cfg_text)
    return parser["info"].get("group", "?")


def load_all_patients() -> dict:
    patients = {}
    for f in sorted(DOPPLER_DIR.glob("patient*.npz")):
        pid = f.stem
        dop = np.load(f)
        proc = np.load(PROC_DIR / f"{pid}.npz")
        split = str(dop["split"])
        patients[pid] = {
            "proj_noisy": dop["proj_noisy"].astype(np.float32),  # (3, Z, 128, 128)
            "target_xy": dop["target_xy"].astype(np.float32),  # (Z, 128, 128, 2)
            "quality_weight": float(dop["quality_weight"]),
            "heart_mask": (proc["es_mask"] > 0).astype(np.float32),  # (Z, 128, 128)
            "ed_mask": proc["ed_mask"],
            "es_mask": proc["es_mask"],
            "spacing": tuple(dop["spacing"]),
            "split": split,
            "group": get_group(pid, split),
        }
    return patients


def stratified_folds(patients: dict, n_folds: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    by_group = {}
    for pid, d in patients.items():
        by_group.setdefault(d["group"], []).append(pid)
    folds = [[] for _ in range(n_folds)]
    for group, pids in by_group.items():
        pids = sorted(pids)
        rng.shuffle(pids)
        for i, pid in enumerate(pids):
            folds[i % n_folds].append(pid)
    return folds


def build_slice_tensors(patients: dict, pids: list):
    xs, ys, ws, ms = [], [], [], []
    for pid in pids:
        d = patients[pid]
        z_dim = d["target_xy"].shape[0]
        proj = d["proj_noisy"]  # (3, Z, 128, 128)
        target = d["target_xy"]  # (Z, 128, 128, 2)
        mask = d["heart_mask"]  # (Z, 128, 128)
        w = d["quality_weight"]
        for z in range(z_dim):
            xs.append(proj[:, z])  # (3, 128, 128)
            ys.append(target[z].transpose(2, 0, 1))  # (2, 128, 128)
            ms.append(mask[z])  # (128, 128)
            ws.append(w)
    X = torch.tensor(np.stack(xs), dtype=torch.float32)
    Y = torch.tensor(np.stack(ys), dtype=torch.float32)
    M = torch.tensor(np.stack(ms), dtype=torch.float32)
    W = torch.tensor(np.array(ws), dtype=torch.float32)
    return X, Y, M, W


def weighted_mse(pred, target, mask, weight):
    # pred, target: (B, 2, H, W); mask: (B, H, W); weight: (B,)
    per_voxel = ((pred - target) ** 2).sum(dim=1)  # (B, H, W)
    masked = per_voxel * mask
    per_sample = masked.sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp(min=1.0)
    return (per_sample * weight).sum() / weight.sum()


def endpoint_error_mm(pred_xy: np.ndarray, true_xy: np.ndarray, mask: np.ndarray) -> float:
    err = np.sqrt(((pred_xy - true_xy) ** 2).sum(axis=-1))  # (Z, H, W)
    if mask.sum() == 0:
        return float("nan")
    return float((err * mask).sum() / mask.sum())


def warped_dice_for_patient(pred_xy: np.ndarray, ed_mask: np.ndarray, es_mask: np.ndarray, spacing: tuple) -> dict:
    z, h, w = pred_xy.shape[:3]
    disp_field = np.zeros((z, h, w, 3), dtype=np.float64)
    disp_field[..., 2] = pred_xy[..., 0]  # dx
    disp_field[..., 1] = pred_xy[..., 1]  # dy
    vec_img = sitk.GetImageFromArray(disp_field, isVector=True)
    vec_img.SetSpacing(spacing)
    transform = sitk.DisplacementFieldTransform(vec_img)
    warped_mask = warp(ed_mask, spacing, es_mask, transform, is_mask=True)
    return dice_per_label(warped_mask, es_mask)


def main() -> None:
    set_all_seeds(SEED)
    print("Loading all patients (ACDC only — M&Ms is not loaded or referenced anywhere in this script)...")
    patients = load_all_patients()
    folds = stratified_folds(patients, N_FOLDS, SEED)

    fold_results = []
    fold_curves = []
    for k in range(N_FOLDS):
        val_pids = folds[k]
        train_pids = [pid for i, f in enumerate(folds) if i != k for pid in f]

        X_train, Y_train, M_train, W_train = build_slice_tensors(patients, train_pids)
        X_val, Y_val, M_val, W_val = build_slice_tensors(patients, val_pids)

        model = SmallMotionCNN()
        if k == 0:
            print(f"Model parameter count: {model.n_params()}")
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        train_losses, val_losses = [], []
        for epoch in range(N_EPOCHS):
            model.train()
            optimizer.zero_grad()
            pred = model(X_train)
            loss = weighted_mse(pred, Y_train, M_train, W_train)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            model.eval()
            with torch.no_grad():
                val_loss = weighted_mse(model(X_val), Y_val, M_val, W_val)
            val_losses.append(val_loss.item())

        # Mean-motion baseline: per-voxel weighted mean of training targets (same grid for all patients)
        with torch.no_grad():
            mean_motion = (Y_train * (M_train * W_train.view(-1, 1, 1)).unsqueeze(1)).sum(dim=0) / (
                (M_train * W_train.view(-1, 1, 1)).sum(dim=0).clamp(min=1e-6)
            ).unsqueeze(0)
        mean_motion_np = mean_motion.numpy()  # (2, 128, 128)

        model.eval()
        for pid in val_pids:
            d = patients[pid]
            proj = torch.tensor(d["proj_noisy"], dtype=torch.float32).permute(1, 0, 2, 3)  # (Z, 3, 128, 128)
            with torch.no_grad():
                pred = model(proj).permute(0, 2, 3, 1).numpy()  # (Z, 128, 128, 2)

            true_xy = d["target_xy"]
            mask = d["heart_mask"]

            zero_pred = np.zeros_like(true_xy)
            mean_pred = np.broadcast_to(mean_motion_np.transpose(1, 2, 0), true_xy.shape)

            model_err = endpoint_error_mm(pred, true_xy, mask)
            zero_err = endpoint_error_mm(zero_pred, true_xy, mask)
            mean_err = endpoint_error_mm(mean_pred, true_xy, mask)

            model_dice = warped_dice_for_patient(pred, d["ed_mask"], d["es_mask"], d["spacing"])
            zero_dice = warped_dice_for_patient(zero_pred.astype(np.float32), d["ed_mask"], d["es_mask"], d["spacing"])

            fold_results.append({
                "fold": k, "patient_id": pid, "group": d["group"],
                "model_endpoint_err_mm": model_err, "zero_endpoint_err_mm": zero_err, "mean_endpoint_err_mm": mean_err,
                "model_mean_dice": float(np.mean(list(model_dice.values()))),
                "zero_mean_dice": float(np.mean(list(zero_dice.values()))),
                "train_loss_first": train_losses[0], "train_loss_last": train_losses[-1],
            })

        print(f"Fold {k}: train loss {train_losses[0]:.4f} -> {train_losses[-1]:.4f}, "
              f"val loss {val_losses[0]:.4f} -> {val_losses[-1]:.4f}, "
              f"{len(val_pids)} val patients evaluated")
        fold_curves.append({"fold": k, "train_losses": train_losses, "val_losses": val_losses})

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, N_FOLDS, figsize=(4 * N_FOLDS, 4), sharey=True)
    for k, ax in enumerate(axes):
        c = fold_curves[k]
        ax.plot(c["train_losses"], label="train")
        ax.plot(c["val_losses"], label="val")
        ax.set_title(f"Fold {k}")
        ax.set_xlabel("epoch")
        ax.legend()
    axes[0].set_ylabel("weighted MSE loss")
    fig.tight_layout()
    curve_path = Path("results/phase5_train_val_curves.png")
    fig.savefig(curve_path, dpi=140)
    plt.close(fig)
    print(f"Train/val curves saved to {curve_path}")

    csv_path = Path("results/phase5_cv_results.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fold_results[0].keys()))
        writer.writeheader()
        writer.writerows(fold_results)

    model_errs = np.array([r["model_endpoint_err_mm"] for r in fold_results])
    zero_errs = np.array([r["zero_endpoint_err_mm"] for r in fold_results])
    mean_errs = np.array([r["mean_endpoint_err_mm"] for r in fold_results])
    model_dices = np.array([r["model_mean_dice"] for r in fold_results])
    zero_dices = np.array([r["zero_mean_dice"] for r in fold_results])

    print(f"\n=== Aggregate over all {len(fold_results)} patients (5-fold CV) ===")
    print(f"Endpoint error (mm): model={model_errs.mean():.3f}+/-{model_errs.std():.3f}, "
          f"zero-motion baseline={zero_errs.mean():.3f}+/-{zero_errs.std():.3f}, "
          f"mean-motion baseline={mean_errs.mean():.3f}+/-{mean_errs.std():.3f}")
    print(f"Warped-mask mean Dice: model={model_dices.mean():.3f}+/-{model_dices.std():.3f}, "
          f"zero-motion baseline={zero_dices.mean():.3f}+/-{zero_dices.std():.3f}")
    print(f"Model beats zero-motion baseline (lower endpoint error): {(model_errs < zero_errs).mean():.1%} of patients")
    print(f"Model beats mean-motion baseline (lower endpoint error): {(model_errs < mean_errs).mean():.1%} of patients")
    print(f"\nResults written to {csv_path}")
    print("\nCONFIRMATION: M&Ms was not loaded, referenced, or evaluated anywhere in this script.")


if __name__ == "__main__":
    main()
