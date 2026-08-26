"""Train the residual visual model (M6) and evaluate one-step + multi-step
open-loop error, nominal vs corrected, on val / test-ID / test-OOD.

Example:
  python scripts/train_residual.py --dataset outputs/datasets/wavept_v1
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wavept.config import config_from_dict, make_nominal
from wavept.manifest import create_run_dir, git_info
from wavept.models.datasets import ACTION_HISTORY_LEN, load_index, load_manifest, load_split
from wavept.models.nominal import ModelState, NominalModel
from wavept.models.residual import (
    CorrectedModel,
    ResidualCorrector,
    build_mlp,
    build_pairs,
    save_checkpoint,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PX = np.array([320.0, 240.0])
HORIZON = 10


def rmse_px(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((y * PX) ** 2, axis=1))))


def train(X, Y, Xv, Yv, hidden, n_hidden, lr, batch, epochs, patience, seed=0):
    import torch

    torch.manual_seed(seed)
    x_mean, x_std = X.mean(0), X.std(0) + 1e-8
    y_mean, y_std = Y.mean(0), Y.std(0) + 1e-8
    Xn, Yn = (X - x_mean) / x_std, (Y - y_mean) / y_std
    Xvn, Yvn = (Xv - x_mean) / x_std, (Yv - y_mean) / y_std
    net = build_mlp(X.shape[1], hidden, n_hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    tX, tY = torch.from_numpy(Xn), torch.from_numpy(Yn)
    vX, vY = torch.from_numpy(Xvn), torch.from_numpy(Yvn)
    rng = np.random.default_rng(seed)
    best_val, best_state, bad, history = np.inf, None, 0, []
    for epoch in range(epochs):
        perm = rng.permutation(len(tX))
        net.train()
        for i in range(0, len(perm), batch):
            sel = torch.from_numpy(perm[i : i + batch])
            opt.zero_grad()
            loss = torch.mean((net(tX[sel]) - tY[sel]) ** 2)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val = float(torch.mean((net(vX) - vY) ** 2))
        history.append(val)
        if val < best_val - 1e-5:
            best_val, best_state, bad = val, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    return net, (x_mean, x_std, y_mean, y_std), epoch + 1, history


def multistep_curve(data, index, split, models, stride=25):
    errors = [[] for _ in range(HORIZON)]
    for _, ep in index[index.split == split].iterrows():
        idx = np.flatnonzero(data["episode_id"] == ep.episode_id)
        model = models[ep.condition]
        for s in range(0, len(idx) - HORIZON - 1, stride):
            i = idx[s]
            if not data["obs_visible"][i]:
                continue
            state = ModelState(
                q=data["q"][i].astype(float), qdot=data["qdot"][i].astype(float),
                uv=data["uv_obs_px"][i].astype(float),
                attitude=data["attitude"][i].astype(float),
                attitude_rate=data["attitude_rate"][i].astype(float),
                queued_actions=data["action_history"][i].reshape(ACTION_HISTORY_LEN, 2).astype(float),
            )
            candidates = data["action"][idx[s] : idx[s] + HORIZON].astype(float)[None]
            uv_pred, _ = model.rollout(state, candidates)
            for k in range(HORIZON):
                j = idx[s] + k
                if data["true_visible_t1"][j]:
                    e = (data["true_uv_norm_t1"][j] - uv_pred[0, k]) * PX
                    errors[k].append(np.linalg.norm(e))
    return np.array([np.mean(e) if e else np.nan for e in errors])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=REPO_ROOT / "outputs" / "datasets" / "wavept_v1")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--n-hidden", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "models" / "residual_v1.pt")
    args = parser.parse_args()

    manifest = load_manifest(args.dataset)
    index = load_index(args.dataset)
    conditions = {n: config_from_dict(c) for n, c in manifest["conditions"].items()}
    nominals = {n: make_nominal(c.sim, c.mismatch, c.scenario) for n, c in conditions.items()}

    splits = ["train", "val", "test_id", "test_ood"]
    data = {s: load_split(args.dataset, s) for s in splits}
    pairs = {s: build_pairs(data[s], index, s, nominals) for s in splits}
    X, Y, _ = pairs["train"]
    Xv, Yv, _ = pairs["val"]
    print(f"pairs: train {len(X)}, val {len(Xv)}, "
          f"test_id {len(pairs['test_id'][0])}, test_ood {len(pairs['test_ood'][0])}")

    t0 = time.perf_counter()
    net, stats, n_epochs, history = train(
        X, Y, Xv, Yv, args.hidden, args.n_hidden, args.lr, args.batch, args.epochs, args.patience
    )
    train_s = time.perf_counter() - t0
    print(f"trained {n_epochs} epochs in {train_s:.0f}s")

    x_mean, x_std, y_mean, y_std = stats
    y_clip = np.percentile(np.abs(Y), 99.5, axis=0) * 3.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "git": git_info(), "dataset": str(args.dataset),
        "hidden": args.hidden, "n_hidden": args.n_hidden,
        "lr": args.lr, "batch": args.batch, "epochs_run": n_epochs,
        "train_seconds": train_s, "n_train_pairs": int(len(X)),
    }
    save_checkpoint(args.out, net, x_mean, x_std, y_mean, y_std,
                    y_clip.astype(np.float32), meta)
    corrector = ResidualCorrector(args.out)

    # ---- one-step evaluation (nominal residual Y vs corrected leftover)
    run_dir = create_run_dir("residual_train")
    rows = []
    for s in splits:
        Xs, Ys, conds = pairs[s]
        if not len(Xs):
            continue
        corr = corrector.predict(Xs)
        for cond in ["ALL"] + sorted(set(conds)):
            m = np.ones(len(Xs), bool) if cond == "ALL" else conds == cond
            rows.append({
                "split": s, "condition": cond, "n": int(m.sum()),
                "nominal_rmse_px": rmse_px(Ys[m]),
                "corrected_rmse_px": rmse_px(Ys[m] - corr[m]),
            })
    table = pd.DataFrame(rows)
    print("\none-step RMSE (px), nominal vs corrected:")
    print(table.to_string(index=False))

    # ---- multi-step open-loop drift, nominal vs corrected
    nom_models = {n: NominalModel(p, rate_decay=1.0) for n, p in nominals.items()}
    cor_models = {n: CorrectedModel(m, corrector) for n, m in nom_models.items()}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    drift = {}
    for s in ["val", "test_id", "test_ood"]:
        cn = multistep_curve(data[s], index, s, nom_models)
        cc = multistep_curve(data[s], index, s, cor_models)
        drift[s] = {"nominal": cn.tolist(), "corrected": cc.tolist()}
        (line,) = ax.plot(range(1, HORIZON + 1), cn, marker="o", linestyle="--",
                          label=f"{s} nominal")
        ax.plot(range(1, HORIZON + 1), cc, marker="s", color=line.get_color(),
                label=f"{s} corrected")
        print(f"10-step drift {s}: nominal {cn[-1]:.1f}px -> corrected {cc[-1]:.1f}px")
    ax.set_xlabel("horizon step (40 ms)"); ax.set_ylabel("mean |error| (px)")
    ax.set_title("Open-loop rollout drift: nominal vs residual-corrected")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(run_dir / "drift_nominal_vs_corrected.png", dpi=120)

    # ---- MPC-shaped inference timing (K=256, N=10)
    d0 = data["val"]
    i = int(np.flatnonzero(d0["obs_visible"])[0])
    state = ModelState(
        q=d0["q"][i].astype(float), qdot=d0["qdot"][i].astype(float),
        uv=d0["uv_obs_px"][i].astype(float), attitude=d0["attitude"][i].astype(float),
        attitude_rate=d0["attitude_rate"][i].astype(float),
        queued_actions=d0["action_history"][i].reshape(ACTION_HISTORY_LEN, 2).astype(float),
    )
    cand = np.clip(np.random.default_rng(0).normal(0, 0.3, (256, HORIZON, 2)), -1, 1)
    some_model = next(iter(cor_models.values()))
    some_model.rollout(state, cand)  # warm-up
    times = []
    for _ in range(20):
        t1 = time.perf_counter()
        some_model.rollout(state, cand)
        times.append((time.perf_counter() - t1) * 1e3)
    print(f"\ncorrected rollout (K=256,N=10): {np.mean(times):.1f} ms mean, "
          f"{np.max(times):.1f} ms max")

    table.to_csv(run_dir / "one_step_rmse.csv", index=False)
    card = {
        **meta,
        "one_step_rmse": table.to_dict("records"),
        "rollout_drift_px": drift,
        "corrected_rollout_ms": {"mean": float(np.mean(times)), "max": float(np.max(times))},
        "val_history": history,
        "limitations": [
            "attitude features are damped-extrapolated at rollout time but true at training time",
            "residual accumulation assumes drift ~ sum of one-step increment errors",
            "OOD conditions (unseen frequencies / calibration) are outside the training distribution",
        ],
    }
    (run_dir / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    (args.out.parent / "residual_v1_card.json").write_text(
        json.dumps(card, indent=2), encoding="utf-8"
    )
    print(f"\ncheckpoint: {args.out}\noutputs: {run_dir}")


if __name__ == "__main__":
    main()
