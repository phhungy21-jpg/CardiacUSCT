"""Train the single final model on ALL 150 ACDC patients, using the exact
frozen architecture/hyperparameters validated by Phase 5's 5-fold CV
(src/run_phase5_cv.py) — no new tuning, just the standard "final fit on all
training data" step after CV has validated the config. This saved model is
what Phase 6 evaluates on M&Ms."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase5_model import SmallMotionCNN  # noqa: E402
from run_phase5_cv import LR, N_EPOCHS, SEED, build_slice_tensors, load_all_patients, weighted_mse  # noqa: E402
from seed import set_all_seeds  # noqa: E402

MODEL_PATH = Path("results/phase5_final_model.pt")


def main() -> None:
    set_all_seeds(SEED)
    print("Loading all 150 ACDC patients (M&Ms not loaded)...")
    patients = load_all_patients()
    all_pids = sorted(patients.keys())

    X, Y, M, W = build_slice_tensors(patients, all_pids)

    model = SmallMotionCNN()
    print(f"Model parameter count: {model.n_params()}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(N_EPOCHS):
        optimizer.zero_grad()
        pred = model(X)
        loss = weighted_mse(pred, Y, M, W)
        loss.backward()
        optimizer.step()
        if epoch == 0 or epoch == N_EPOCHS - 1:
            print(f"epoch {epoch}: loss {loss.item():.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Final model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
