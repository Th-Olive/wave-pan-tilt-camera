"""M5 tests: dataset round-trip, split leakage, timeline alignment of the
nominal one-step prediction."""

import numpy as np
import pytest

from wavept.models.collection import (
    ID_CONDITIONS,
    OOD_CONDITIONS,
    collect,
)
from wavept.models.datasets import ACTION_HISTORY_LEN, load_index, load_manifest, load_split

TINY_PLAN = {
    "train": {c: {"ou": 1, "pgain": 1} for c in ID_CONDITIONS},
    "val": {c: {"ou": 1} for c in ID_CONDITIONS},
    "test_ood": {c: {"ou": 1} for c in OOD_CONDITIONS},
}


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("data") / "tiny"
    collect(out, plan=TINY_PLAN, duration=3.0)  # 75 steps per episode
    return out


def test_round_trip_and_counts(dataset):
    index = load_index(dataset)
    manifest = load_manifest(dataset)
    assert manifest["n_episodes"] == len(index) == 11
    train = load_split(dataset, "train")
    assert train["q"].shape[1] == 2
    assert train["action_history"].shape[1] == ACTION_HISTORY_LEN * 2
    n_train = index[index.split == "train"].n_steps.sum()
    assert len(train["t"]) == n_train


def test_no_split_leakage(dataset):
    index = load_index(dataset)
    by_split = {s: g for s, g in index.groupby("split")}
    # disjoint episode ids and seeds across splits
    for a in by_split:
        for b in by_split:
            if a >= b:
                continue
            assert not set(by_split[a].episode_id) & set(by_split[b].episode_id)
            assert not set(by_split[a].seed) & set(by_split[b].seed)
    # OOD conditions never appear in train/val
    assert set(by_split["train"].condition) <= set(ID_CONDITIONS)
    assert set(by_split["test_ood"].condition) <= set(OOD_CONDITIONS)
    assert not set(by_split["train"].condition) & set(by_split["test_ood"].condition)


def test_action_history_reconstruction(dataset):
    train = load_split(dataset, "train")
    ep = train["episode_id"] == train["episode_id"][0]
    actions = train["action"][ep]
    hist = train["action_history"][ep].reshape(-1, ACTION_HISTORY_LEN, 2)
    L = ACTION_HISTORY_LEN
    np.testing.assert_array_equal(hist[0], np.zeros((L, 2)))  # zero-padded start
    for i in [L, L + 5, len(actions) - 1]:
        np.testing.assert_array_equal(hist[i], actions[i - L : i])


def test_one_step_prediction_alignment(dataset):
    """On the calm condition (mild waves, zero mismatch) the nominal one-step
    prediction must be within ~1 px of the true next position — this pins the
    timeline alignment (obs delay, action replay, label indexing)."""
    index = load_index(dataset)
    train = load_split(dataset, "train")
    calm_ids = set(index[(index.split == "train") & (index.condition == "calm")].episode_id)
    m = (
        np.isin(train["episode_id"], list(calm_ids))
        & train["obs_visible"]
        & train["true_visible_t1"]
        & np.isfinite(train["pred_uv_norm_t1"]).all(axis=1)
    )
    err_px = np.linalg.norm(
        (train["true_uv_norm_t1"][m] - train["pred_uv_norm_t1"][m]) * [320.0, 240.0], axis=1
    )
    assert len(err_px) > 100
    assert np.sqrt(np.mean(err_px**2)) < 1.0


def test_rough_condition_has_systematic_error(dataset):
    """Mismatch + waves must produce clearly larger nominal error than calm —
    otherwise there is nothing for the residual model (M6) to learn."""
    index = load_index(dataset)
    train = load_split(dataset, "train")
    conds = index.set_index("episode_id").condition

    def rms(cond):
        ids = set(conds[conds == cond].index)
        m = (
            np.isin(train["episode_id"], list(ids))
            & train["obs_visible"]
            & train["true_visible_t1"]
            & np.isfinite(train["pred_uv_norm_t1"]).all(axis=1)
        )
        e = (train["true_uv_norm_t1"][m] - train["pred_uv_norm_t1"][m]) * [320.0, 240.0]
        return np.sqrt(np.mean(np.linalg.norm(e, axis=1) ** 2))

    assert rms("rough") > 2.0 * rms("calm")
