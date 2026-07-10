"""The "reverse engineer this" experiment, per user: does a SIMPLE CNN,
trained end-to-end on one beating heart's raw (matched-filter-
compressed, but otherwise unprocessed) acoustic recordings, predict
something genuinely indicative when tested on a completely different,
never-seen ("unleaked") patient's beating heart?

Bypasses EVERY hand-engineered candidate-generation/rescoring method
this project has built (runs -08 through -38: peak detection, DAS
imaging, speckle scoring, joint optimization) -- the only fixed,
physics-justified preprocessing step is matched filtering (pulse
compression against the KNOWN transmit waveform, a standard front end,
not a learned or hand-picked feature). Everything after that is left
entirely to the network.

TRAIN: patient001 (8 phases x 18 angles = 144 examples, ~1.6% real
contraction). TEST: patient023 (144 examples, ~18% real contraction --
a genuinely harder, different-distribution held-out patient, never
touched during training).

No ML framework available in this project's venv (torch/tensorflow/
sklearn/flax/optax all absent) -- implemented directly in raw JAX:
2 strided 1D-conv layers + global average pooling + 1 dense layer,
trained via a hand-written Adam optimizer on `jax.grad`.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import random

SEED = 42
N_EPOCHS = 400
LR = 3e-3
WEIGHT_DECAY = 1e-3


def init_params(key, input_len):
    k1, k2, k3 = random.split(key, 3)
    conv1_w = random.normal(k1, (15, 1, 8)) * jnp.sqrt(2.0 / (15 * 1))
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
    # x: (N, W, 1)
    h = jnp.maximum(conv1d(x, params["conv1_w"], params["conv1_b"], stride=4), 0.0)
    h = jnp.maximum(conv1d(h, params["conv2_w"], params["conv2_b"], stride=4), 0.0)
    h = jnp.mean(h, axis=1)  # global average pool -> (N, 16)
    out = h @ params["dense_w"] + params["dense_b"]  # (N, 1)
    return out[:, 0]


def loss_fn(params, x, y):
    pred = forward(params, x)
    mse = jnp.mean((pred - y) ** 2)
    l2 = sum(jnp.sum(v ** 2) for k, v in params.items() if k.endswith("_w"))
    return mse + WEIGHT_DECAY * l2


def adam_init(params):
    return {"m": jax.tree_util.tree_map(jnp.zeros_like, params),
            "v": jax.tree_util.tree_map(jnp.zeros_like, params), "t": 0}


def adam_step(params, grads, state, lr, b1=0.9, b2=0.999, eps=1e-8):
    t = state["t"] + 1
    m = jax.tree_util.tree_map(lambda m_, g: b1 * m_ + (1 - b1) * g, state["m"], grads)
    v = jax.tree_util.tree_map(lambda v_, g: b2 * v_ + (1 - b2) * (g ** 2), state["v"], grads)
    m_hat = jax.tree_util.tree_map(lambda m_: m_ / (1 - b1 ** t), m)
    v_hat = jax.tree_util.tree_map(lambda v_: v_ / (1 - b2 ** t), v)
    new_params = jax.tree_util.tree_map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, m_hat, v_hat)
    return new_params, {"m": m, "v": v, "t": t}


if __name__ == "__main__":
    d = np.load("results/beating_heart_dataset.npz")
    X_train, y_train = d["X_train"], d["y_train"]
    X_test, y_test = d["X_test"], d["y_test"]

    # normalize X feature-wise and y, using TRAIN statistics only (no test leakage)
    x_mu, x_sd = X_train.mean(), X_train.std() + 1e-12
    y_mu, y_sd = y_train.mean(), y_train.std() + 1e-12
    Xtr = ((X_train - x_mu) / x_sd)[..., None]
    Xte = ((X_test - x_mu) / x_sd)[..., None]
    ytr = (y_train - y_mu) / y_sd
    yte_true = y_test  # keep raw for final reporting

    print(f"train: X={Xtr.shape}, y mean={y_mu:.2f} std={y_sd:.2f} cells")
    print(f"test:  X={Xte.shape}")

    key = random.PRNGKey(SEED)
    params = init_params(key, Xtr.shape[1])
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
    train_rmse_mm = np.sqrt(np.mean((pred_train - y_train) ** 2)) * 0.1  # DX_M=0.1mm/cell
    test_corr = np.corrcoef(pred_test, yte_true)[0, 1]
    test_rmse_mm = np.sqrt(np.mean((pred_test - yte_true) ** 2)) * 0.1

    print(f"\n--- FINAL RESULT ---")
    print(f"  TRAIN (patient001, seen):      corr={train_corr:.3f}, RMSE={train_rmse_mm:.3f}mm")
    print(f"  TEST  (patient023, UNLEAKED):  corr={test_corr:.3f}, RMSE={test_rmse_mm:.3f}mm")
    print(f"\n  (compare: this project's best hand-engineered inner-boundary result on "
          f"patient023, run -38 joint optimization: corr=0.416-0.566, RMSE=1.2-2.1mm)")

    if test_corr > 0.4:
        print("  -> The network generalizes: real, indicative signal exists beyond patient001.")
    elif test_corr > 0.15:
        print("  -> Weak generalization -- some signal, but not strong.")
    else:
        print("  -> No generalization: the network likely memorized patient001-specific patterns "
              "rather than learning something transferable.")

    np.savez("results/beating_heart_cnn_predictions.npz",
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

    axes[1].scatter(yte_true, pred_test, alpha=0.6, color="C1", label=f"test, corr={test_corr:.2f}")
    lims2 = [min(yte_true.min(), pred_test.min()), max(yte_true.max(), pred_test.max())]
    axes[1].plot(lims2, lims2, "k--", alpha=0.5)
    axes[1].set_xlabel("true inner radius (cells)")
    axes[1].set_ylabel("CNN predicted radius (cells)")
    axes[1].set_title(f"TEST (patient023, UNLEAKED)\nRMSE={test_rmse_mm:.2f}mm")
    axes[1].legend(fontsize=8)

    fig.suptitle("Simple end-to-end CNN: raw acoustic recording -> inner-boundary radius")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/phase1_beating_heart_cnn.png", dpi=140)
    print("\nSaved results/figures/phase1_beating_heart_cnn.png")
