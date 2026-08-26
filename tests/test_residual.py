"""M6 tests: residual model plumbing (torch), one-step batch consistency,
zero-residual equivalence, rollout contract and budget."""

import dataclasses
import time
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wavept.config import load_config, make_nominal
from wavept.models.nominal import ModelState, NominalModel
from wavept.models.residual import (
    N_FEATURES,
    CorrectedModel,
    build_mlp,
    make_features,
    one_step_nominal_batch,
)

DYN = Path(__file__).resolve().parents[1] / "configs" / "sim_dynamics.yaml"


@pytest.fixture()
def nominal():
    config = load_config(DYN)
    return make_nominal(config.sim, config.mismatch, config.scenario)


def delayless(nominal):
    return dataclasses.replace(
        nominal, pan_tilt=dataclasses.replace(nominal.pan_tilt, n_cmd_delay=0, n_obs_delay=0)
    )


def test_one_step_batch_matches_rollout(nominal):
    """The vectorized aligned one-step predictor must equal a delayless
    NominalModel single-step rollout."""
    nom0 = delayless(nominal)
    model = NominalModel(nom0, rate_decay=1.0)
    rng = np.random.default_rng(0)
    n = 16
    q = np.column_stack([rng.uniform(0.3, 0.6, n), rng.uniform(0.25, 0.45, n)])
    qdot = rng.normal(0.0, 0.5, (n, 2))
    uv = rng.uniform(-0.5, 0.5, (n, 2))
    att = rng.normal(0.0, 0.1, (n, 2))
    rate = rng.normal(0.0, 0.3, (n, 2))
    a = rng.uniform(-1, 1, (n, 2))
    pred, valid = one_step_nominal_batch(q, qdot, uv, att, att + rate * nom0.dt, a, nom0)
    assert valid.all()
    for i in range(n):
        cam = nom0.camera
        uv_px = np.array([(uv[i, 0] + 1) * cam.width / 2, (uv[i, 1] + 1) * cam.height / 2])
        state = ModelState(q[i], qdot[i], uv_px, att[i], rate[i], np.zeros((0, 2)))
        uv_roll, ok = model.rollout(state, a[i][None, None])
        assert ok[0, 0]
        np.testing.assert_allclose(pred[i], uv_roll[0, 0], atol=1e-10)


def test_zero_residual_equals_nominal(nominal):
    model = NominalModel(nominal, rate_decay=1.0)

    class ZeroCorrector:
        def __call__(self, q, qdot, uv, att, rate, a, ap):
            return np.zeros((len(q), 2))

    corrected = CorrectedModel(model, ZeroCorrector())
    rng = np.random.default_rng(1)
    state = ModelState(
        q=np.array([0.45, 0.35]), qdot=np.array([0.2, -0.1]),
        uv=np.array([300.0, 250.0]), attitude=np.array([0.05, -0.03]),
        attitude_rate=np.array([0.2, 0.1]),
        queued_actions=rng.uniform(-1, 1, (3, 2)),
    )
    cand = np.clip(rng.normal(0, 0.4, (8, 10, 2)), -1, 1)
    uv_n, ok_n = model.rollout(state, cand)
    uv_c, ok_c = corrected.rollout(state, cand)
    np.testing.assert_allclose(uv_c, uv_n, atol=1e-12)
    np.testing.assert_array_equal(ok_c, ok_n)


def test_mlp_overfits_tiny_set():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(64, N_FEATURES)).astype(np.float32)
    Y = np.column_stack([np.sin(X[:, 0]) * 0.1, X[:, 1] * 0.05]).astype(np.float32)
    torch.manual_seed(0)
    net = build_mlp(hidden=64)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    tX, tY = torch.from_numpy(X), torch.from_numpy(Y)
    for _ in range(600):
        opt.zero_grad()
        loss = torch.mean((net(tX) - tY) ** 2)
        loss.backward()
        opt.step()
    assert float(loss) < 1e-4


def test_corrected_rollout_shape_and_budget(nominal, tmp_path):
    from wavept.models.residual import ResidualCorrector, save_checkpoint

    net = build_mlp()
    path = tmp_path / "r.pt"
    save_checkpoint(
        path, net,
        np.zeros(N_FEATURES, np.float32), np.ones(N_FEATURES, np.float32),
        np.zeros(2, np.float32), np.ones(2, np.float32) * 0.01,
        np.full(2, 0.05, np.float32), {"hidden": 64, "n_hidden": 2},
    )
    corrected = CorrectedModel(NominalModel(nominal), ResidualCorrector(path))
    rng = np.random.default_rng(3)
    state = ModelState(
        q=np.array([0.45, 0.35]), qdot=np.zeros(2), uv=np.array([320.0, 240.0]),
        attitude=np.zeros(2), attitude_rate=np.zeros(2),
        queued_actions=np.zeros((3, 2)),
    )
    cand = np.clip(rng.normal(0, 0.3, (256, 10, 2)), -1, 1)
    uv, ok = corrected.rollout(state, cand)  # warm-up + shape
    assert uv.shape == (256, 10, 2) and ok.shape == (256, 10)
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        corrected.rollout(state, cand)
        times.append((time.perf_counter() - t0) * 1e3)
    assert np.mean(times) < 40.0, f"corrected rollout {np.mean(times):.1f} ms"


def test_make_features_broadcast():
    f = make_features(
        np.zeros((5, 2)), np.zeros(2), np.zeros((5, 2)), np.zeros(2),
        np.zeros(2), np.zeros((5, 2)), np.zeros((5, 2))
    )
    assert f.shape == (5, N_FEATURES)
    assert f.dtype == np.float32
