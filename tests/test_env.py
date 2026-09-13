"""Fast env contract test (no torch): `just test`."""

import numpy as np

from d3il_flow.env import ACTION_BOUND, AvoidingChunkEnv, EnvConfig, Normalizer, obstacle_xy


def test_chunk_env_contract():
    env = AvoidingChunkEnv(EnvConfig(ep_len=8, act_steps=4), Normalizer.default())
    obs = env.reset()
    assert obs.shape == (4,) and obs.dtype == np.float32 and np.all(np.abs(obs) <= 1.0)
    assert np.allclose(obs[:2], obs[2:], atol=0.05), (
        "at reset the setpoint equals the measured EE pos"
    )
    obs, _r, done, info = env.step(np.zeros((4, 2), np.float32))
    assert not done and info["t"] == 4
    obs, _r, done, info = env.step(np.ones((4, 2), np.float32))  # max speed towards +x, +y
    assert done and info["t"] == 8 and info["ee_traj"].shape == (9, 2)
    assert set(info) >= {"success", "collision", "reached_goal", "mode_encoding", "ee_pos"}
    assert info["mode_encoding"].shape == (9,)
    env.close()


def test_normalizer_roundtrip():
    n = Normalizer.default()
    a = np.random.uniform(-ACTION_BOUND, ACTION_BOUND, (16, 2)).astype(np.float32)
    assert np.allclose(n.unact(n.act(a)), a, atol=1e-6)
    assert obstacle_xy().shape == (6, 2)
