"""
Stackelberg environment wrapper, CPA variant.

This is a sibling of `env_stackelberg.py` that swaps in 2D Closest-Point-of-
Approach (CPA) leader selection in place of the 1D range-rate TTC used by
`CCEnv`. The original `env_stackelberg.py` is left untouched so we can run
both side-by-side and compare.

Composition:

    MetaDrive env
        + CCEnv mixin          <-- distance map + range-rate leader info
        + CPACCEnv override    <-- overwrites the leader_* info fields with
                                   CPA-based ones (range-rate fields are
                                   computed but immediately discarded)
        + RLlib MultiAgentEnv adapter

The algorithm side (`algo_stackelberg.py`) only reads
    info[agent_id]["leader_id"]
    info[agent_id]["leader_ttc"]
    info[agent_id]["ego_ttc_to_conflict"]
so the contract is identical -- only the formula behind those fields changes.

Config knobs added by this variant (set via env_config):
    cpa_r_safe         : float, conflict gate on miss distance (m). Default 3.0
    cpa_neighbor_radius: float, coarse current-distance prune (m). Default 10.0

Diagnostics added to info (for analysis only, the algo ignores them):
    leader_d_min : CPA miss distance of the chosen leader, +inf if none.
"""

import numpy as np

from copo.torch_copo.utils.env_wrappers import (
    CCEnv,
    get_rllib_compatible_env,
)
from copo.torch_stackelberg.ttc_cpa import compute_pairwise_cpa_leaders


class CPACCEnv(CCEnv):
    """
    CCEnv + CPA leader override.

    Strategy: let CCEnv.step run normally (it already fills distance map,
    neighbours, comm channel, and the range-rate leader fields), then in
    this subclass we recompute the leader fields with CPA and overwrite
    them in `info`. This keeps CCEnv untouched and minimises code drift.
    """

    @classmethod
    def default_config(cls):
        cfg = super(CPACCEnv, cls).default_config()
        cfg.update(
            dict(
                cpa_r_safe=3.0,
                cpa_neighbor_radius=10.0,
            )
        )
        return cfg

    def step(self, actions):
        o, r, d, i = super(CPACCEnv, self).step(actions)

        agent_states = {}
        vehicles = self._safe_vehicles()
        for agent_id in o.keys():
            vehicle = vehicles.get(agent_id, None)
            if vehicle is None:
                continue
            position = np.array(vehicle.position, dtype=np.float32)
            if hasattr(vehicle, "velocity"):
                velocity = np.array(vehicle.velocity, dtype=np.float32)
            else:
                heading = vehicle.heading_theta
                speed = vehicle.speed
                velocity = np.array(
                    [speed * np.cos(heading), speed * np.sin(heading)],
                    dtype=np.float32,
                )
            agent_states[agent_id] = {"position": position, "velocity": velocity}

        r_safe = float(self.config.get("cpa_r_safe", 3.0))
        nbr_radius = float(self.config.get("cpa_neighbor_radius", 10.0))

        leader_infos = compute_pairwise_cpa_leaders(
            agent_states,
            neighbor_radius=nbr_radius,
            r_safe=r_safe,
        )
        for agent_id, leader_info in leader_infos.items():
            if agent_id not in i:
                i[agent_id] = {}
            # Overwrite the range-rate leader fields with CPA ones.
            i[agent_id]["leader_id"] = leader_info["leader_id"]
            i[agent_id]["leader_ttc"] = leader_info["leader_ttc"]
            i[agent_id]["ego_ttc_to_conflict"] = leader_info["ego_ttc_to_conflict"]
            # Diagnostic-only: miss distance of the chosen leader pairing.
            i[agent_id]["leader_d_min"] = leader_info["leader_d_min"]

        return o, r, d, i


def get_stackelberg_cpa_env(env_class):
    """
    Build a Stackelberg-flavoured RLlib MultiAgentEnv with CPA leader selection.

    Mirrors `get_stackelberg_env` but uses `CPACCEnv` in place of plain
    `CCEnv`, and registers under a different name so it cannot collide with
    the range-rate variant.
    """
    base_name = env_class.__name__

    class _StackelbergCPAEnv(CPACCEnv, env_class):
        @classmethod
        def default_config(cls):
            cfg = super(_StackelbergCPAEnv, cls).default_config()
            cfg["communication"]["comm_method"] = "none"
            return cfg

    name = "StackelbergCPA{}".format(base_name)
    _StackelbergCPAEnv.__name__ = name
    _StackelbergCPAEnv.__qualname__ = name

    return get_rllib_compatible_env(_StackelbergCPAEnv)
