import numpy as np

from fdrl.conditions import Condition, seed_for
from fdrl.data.manifest import DatasetManifest, load_training_arrays
from fdrl.data.schema import EpisodeSummary
from fdrl.data.store import EpisodeStore
from fdrl.episodes import to_records
from fdrl.expert.scripted import ScriptedExpert
from fdrl.sim.base import make_backend
from fdrl.sim.rollout import rollout


def test_roundtrip_bit_exact(tmp_path, cfg, grid):
    be = make_backend(cfg, 8)
    ex = ScriptedExpert(cfg.expert, cfg.task)
    items = [(Condition(0.02 * i - 0.05, 0.05, 0.1, 0.5), seed_for("acq", 0, 0, i)) for i in range(8)]
    res = rollout(be, items, lambda o, p: ex.act(p), cfg.task)
    recs = to_records(res, split="acq", source="expert", grid=grid, task=cfg.task, fail_cfg=cfg.failures)
    store = EpisodeStore(tmp_path)
    store.write([s for s, _ in recs], {s.episode_id: t for s, t in recs})
    idx = store.index()
    assert len(idx) == 8
    for s, tr in recs:
        back = store.load_trajectory(s.traj_ref)
        np.testing.assert_array_equal(back.obs, tr.obs)
        np.testing.assert_array_equal(back.actions, tr.actions)
        for k in tr.privileged:
            np.testing.assert_array_equal(back.privileged[k], tr.privileged[k])
        s2 = store.summaries([s.episode_id])[0]
        assert s2.condition == s.condition and s2.success == s.success and s2.failure_flags == s.failure_flags

    ids = [s.episode_id for s, _ in recs if s.accepted]
    m = DatasetManifest.create("d", ids)
    m.save(tmp_path / "m.json")
    assert DatasetManifest.load(tmp_path / "m.json") == m
    arr = load_training_arrays(store, m, grid)
    assert arr.obs.shape[0] == sum(len(t.actions) for s, t in recs if s.accepted)
    arr_l = load_training_arrays(store, m, grid, include_latent=True)
    assert arr_l.obs.shape[1] == arr.obs.shape[1] + 1


def test_manifest_hash_and_extend():
    a = DatasetManifest.create("a", ["x", "y"])
    b = DatasetManifest.create("b", ["y", "x"])
    assert a.sha256 == b.sha256
    c = a.extend("c", ["z"])
    assert c.parent == a.sha256 and len(c) == 3


def test_summary_row_roundtrip():
    s = EpisodeSummary(split="val", seed=2_000_001, condition=Condition(0.1, -0.2, 0.3, 0.4), cell=5,
                       source="policy", success=False, length=250, sim_steps_charged=250, accepted=False,
                       failure_label="F4", failure_flags=["F2", "F4"], failure_t=120, diag={"max_slip": 0.02})
    s2 = EpisodeSummary.from_row(s.to_row())
    assert s2.failure_flags == ["F2", "F4"] and s2.diag["max_slip"] == 0.02 and s2.round is None
