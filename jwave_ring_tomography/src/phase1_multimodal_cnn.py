"""Trains the SAME architecture as `phase1_beating_heart_cnn.py`, but on
the multi-modal (reflection + previous-phase/motion) 2-channel dataset,
for a direct, apples-to-apples comparison: does fusing the motion
channel in at the input improve generalization to patient023 (held out
entirely) over the original single-channel (reflection-only) CNN?
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import random

from phase1_beating_heart_cnn import adam_init, adam_step, N_EPOCHS, LR, WEIGHT_DECAY, SEED

N_CHANNELS = 2


def init_params(key, n_channels):
    k1, k2, k3 = random.split(key, 3)
    conv1_w = random.normal(k1, (15, n_channels, 8)) * jnp.sqrt(2.0 / (15 * n_channels))
    conv1_b = jnp.zeros(8)
    conv2_w = random.normal(k2, (9, 8, 16)) * jnp.sqrt(2.0 / (9 * 8))
    conv2_b = jnp.zeros(16)
    dense_w = random.normal(k3, (16, 1)) * jnp.sqrt(2.0 / 16)
    dense_b = jnp.zeros(1)
    return {"conv1_w": conv1_w, "conv1_b": conv1_b, "conv2_w": conv2_w, "conv2_b": conv2_b,
            "dense_w": dense_w, "dense_b": dense_b}


def conv1d(x, w, b, stride):
    y = jax.lax.conv_general_dilated(
        x, w, window_strides=(stride,), padding="VALID",
        dimension_numbers=("NWC", "WIO", "NWC"))
    return y + b


def forward(params, x):
    h = jnp.maximum(conv1d(x, params["conv1_w"], params["conv1_b"], stride=4), 0.0)
    h = jnp.maximum(conv1d(h, params["conv2_w"], params["conv2_b"], stride=4), 0.0)
    h = jnp.mean(h, axis=1)
    out = h @ params["dense_w"] + params["dense_b"]
    return out[:, 0]


def loss_fn(params, x, y):
    pred = forward(params, x)
    mse = jnp.mean((pred - y) ** 2)
    l2 = sum(jnp.sum(v ** 2) for k, v in params.items() if k.endswith("_w"))
    return mse + WEIGHT_DECAY * l2


if __name__ == "__main__":
    d = np.load("results/multimodal_heart_dataset.npz")
    X_train, y_train = d["X_train"], d["y_train"]
    X_test, y_test = d["X_test"], d["y_test"]

    x_mu, x_sd = X_train.mean(), X_train.std() + 1e-12
    y_mu, y_sd = y_train.mean(), y_train.std() + 1e-12
    Xtr = (X_train - x_mu) / x_sd
    Xte = (X_test - x_mu) / x_sd
    ytr = (y_train - y_mu) / y_sd

    print(f"train: X={Xtr.shape}, y mean={y_mu:.2f} std={y_sd:.2f} cells")
    print(f"test:  X={Xte.shape}")

    key = random.PRNGKey(SEED)
    params = init_params(key, N_CHANNELS)
    opt_state = adam_init(params)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    forward_jit = jax.jit(forward)

    for epoch in range(N_EPOCHS):
        loss, grads = grad_fn(params, jnp.array(Xtr), jnp.array(ytr))
        params, opt_state = adam_step(params, grads, opt_state, LR)
        if epoch % 50 == 0 or epoch == N_EPOCHS - 1:
            pred_train = np.array(forward_jit(params, jnp.array(Xtr))) * y_sd + y_mu
            train_corr = np.corrcoef(pred_train, y_train)[0, 1]
            print(f"  epoch {epoch:4d}: loss={loss:.4f}, train corr={train_corr:.3f}")

    pred_train = np.array(forward_jit(params, jnp.array(Xtr))) * y_sd + y_mu
    pred_test = np.array(forward_jit(params, jnp.array(Xte))) * y_sd + y_mu

    train_corr = np.corrcoef(pred_train, y_train)[0, 1]
    train_rmse_mm = np.sqrt(np.mean((pred_train - y_train) ** 2)) * 0.1
    test_corr = np.corrcoef(pred_test, y_test)[0, 1]
    test_rmse_mm = np.sqrt(np.mean((pred_test - y_test) ** 2)) * 0.1

    lo, hi = y_train.min(), y_train.max()
    in_range = (y_test >= lo) & (y_test <= hi)
    corr_in = np.corrcoef(pred_test[in_range], y_test[in_range])[0, 1]
    corr_out = np.corrcoef(pred_test[~in_range], y_test[~in_range])[0, 1] if (~in_range).sum() > 2 else float("nan")

    print(f"\n--- FINAL RESULT (multi-modal: reflection + previous-phase/motion, 2-channel) ---")
    print(f"  TRAIN (patient001, seen):      corr={train_corr:.3f}, RMSE={train_rmse_mm:.3f}mm")
    print(f"  TEST  (patient023, UNLEAKED):  corr={test_corr:.3f}, RMSE={test_rmse_mm:.3f}mm")
    print(f"  TEST, within train's y-range ({in_range.sum()}/{len(y_test)}):     corr={corr_in:.3f}")
    print(f"  TEST, outside train's y-range ({(~in_range).sum()}/{len(y_test)}): corr={corr_out:.3f}")
    print(f"\n  (compare: SINGLE-channel CNN, run this session: TEST corr=0.233 overall, "
          f"in-range corr=0.114, out-of-range corr=0.509)")

    if test_corr > 0.233 + 0.1:
        print("  -> Multi-modal fusion IMPROVES generalization over the single (reflection-only) channel.")
    elif test_corr < 0.233 - 0.1:
        print("  -> Multi-modal fusion is WORSE than the single channel at this setting.")
    else:
        print("  -> No clear difference from fusing in the motion channel this way.")

    np.savez("results/multimodal_heart_cnn_predictions.npz",
             pred_train=pred_train, y_train=y_train, pred_test=pred_test, y_test=y_test)

    from matplotlib import pyplot as plt
    import labels
    import os
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    axes[0].scatter(y_train, pred_train, alpha=0.6, label=f"train, corr={train_corr:.2f}")
    lims = [min(y_train.min(), pred_train.min()), max(y_train.max(), pred_train.max())]
    axes[0].plot(lims, lims, "k--", alpha=0.5)
    axes[0].set_xlabel("true inner radius (cells)")
    axes[0].set_ylabel("CNN predicted radius (cells)")
    axes[0].set_title(f"TRAIN (patient001, seen)\nRMSE={train_rmse_mm:.2f}mm")
    axes[0].legend(fontsize=8)

    axes[1].scatter(y_test, pred_test, alpha=0.6, color="C1", label=f"test, corr={test_corr:.2f}")
    lims2 = [min(y_test.min(), pred_test.min()), max(y_test.max(), pred_test.max())]
    axes[1].plot(lims2, lims2, "k--", alpha=0.5)
    axes[1].set_xlabel("true inner radius (cells)")
    axes[1].set_ylabel("CNN predicted radius (cells)")
    axes[1].set_title(f"TEST (patient023, UNLEAKED)\nRMSE={test_rmse_mm:.2f}mm")
    axes[1].legend(fontsize=8)

    fig.suptitle("Multi-modal CNN: reflection + previous-phase/motion (2-channel) -> inner-boundary radius")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/phase1_multimodal_cnn.png", dpi=140)
    print("\nSaved results/figures/phase1_multimodal_cnn.png")
