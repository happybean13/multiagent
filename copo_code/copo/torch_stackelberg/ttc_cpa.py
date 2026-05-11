"""
2D Closest-Point-of-Approach (CPA) leader selection.

This is a drop-in replacement for `compute_pairwise_ttc_leaders` in
`copo.torch_copo.utils.env_wrappers` that uses a proper 2D constant-velocity
miss-distance check, instead of the 1D range-rate ("bearing line") TTC.

Per-pair formulas (let dp = p_i - p_j, dv = v_i - v_j):

    t*   = - (dp . dv) / |dv|^2                # time of closest approach
    dmin = | dp + t* * dv |                    # miss distance at t*

We declare a pairwise conflict iff:
    |dv|^2 > eps        (agents actually have a relative motion direction)
    t*    > 0           (closest approach is in the future)
    dmin  < r_safe      (they would pass within the safety radius)

In that case TTC_ij := t*. Otherwise TTC_ij := +inf.

Leader rule (kept identical to the range-rate version, only the per-pair TTC
formula changes, so the rest of the Stackelberg algorithm sees the same
interface and semantics):

    For each agent i, let min_ttc[i] = min_j TTC_ij and j* = argmin_j TTC_ij.
    Assign leader_id[i] = j*  iff  min_ttc[i] > min_ttc[j*],
    i.e. j* reaches conflict strictly earlier than i does, so j* "goes first"
    and i must follow.

Notes
-----
* The neighbour radius filter (default 10 m) is kept as a coarse prune so we
  do not run CPA against every car in the scene -- pairs farther than this in
  *current* Euclidean distance are skipped.
* CPA is symmetric in (i, j): TTC_ij == TTC_ji and dmin_ij == dmin_ji, exactly
  like the original range-rate TTC. The leader rule therefore behaves the
  same way; only the *which pair fires* changes.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


_EPS_VEL_SQ = 1e-8  # guard for |dv|^2 ~ 0 (parallel motion at same speed)


def _cpa_pair(p_i, v_i, p_j, v_j, r_safe):
    """
    CPA test for a single ordered pair (i, j). Returns (ttc, d_min).

    ttc is +inf iff the pair has no future near-miss inside r_safe.
    """
    dp = p_i - p_j
    dv = v_i - v_j
    dv_sq = float(np.dot(dv, dv))
    if dv_sq < _EPS_VEL_SQ:
        # Effectively parallel motion at identical speed: no future contact.
        return np.inf, float(np.linalg.norm(dp))

    t_star = -float(np.dot(dp, dv)) / dv_sq
    if t_star <= 0.0:
        # Closest approach is now or in the past -> they are diverging.
        return np.inf, float(np.linalg.norm(dp))

    miss = dp + t_star * dv
    d_min = float(np.linalg.norm(miss))
    if d_min >= r_safe:
        return np.inf, d_min

    return t_star, d_min


def compute_pairwise_cpa_leaders(
    agent_states: Dict[str, Dict[str, np.ndarray]],
    neighbor_radius: float = 10.0,
    r_safe: float = 3.0,
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    2D CPA analogue of `compute_pairwise_ttc_leaders`.

    Parameters
    ----------
    agent_states : dict
        { agent_id: {"position": np.ndarray shape (2,),
                     "velocity": np.ndarray shape (2,)} }
    neighbor_radius : float
        Coarse current-distance prune (m). Pairs farther than this in
        instantaneous Euclidean distance are skipped (matches the original
        range-rate function's 10 m default).
    r_safe : float
        Safety radius (m) used as the conflict gate on the CPA miss distance.

    Returns
    -------
    dict
        {
          agent_id: {
            "leader_id": str or None,
            "leader_ttc": float,            # +inf if no leader
            "ego_ttc_to_conflict": float,   # +inf if no future near-miss
            "leader_d_min": float,          # +inf if no leader (diagnostic)
          }
        }
    """
    agent_ids = list(agent_states.keys())
    n = len(agent_ids)

    min_ttc = {aid: np.inf for aid in agent_ids}
    min_d_min = {aid: np.inf for aid in agent_ids}
    closest_conflict_agent = {aid: None for aid in agent_ids}

    # 1) Pairwise CPA TTC, with coarse radius prune.
    for a_i in range(n):
        i = agent_ids[a_i]
        p_i = agent_states[i]["position"]
        v_i = agent_states[i]["velocity"]
        for a_j in range(n):
            if a_i == a_j:
                continue
            j = agent_ids[a_j]
            p_j = agent_states[j]["position"]
            v_j = agent_states[j]["velocity"]

            cur_dist = float(np.linalg.norm(p_j - p_i))
            if cur_dist > neighbor_radius or cur_dist < 1e-6:
                continue

            ttc, d_min = _cpa_pair(p_i, v_i, p_j, v_j, r_safe)
            if ttc < min_ttc[i]:
                min_ttc[i] = ttc
                min_d_min[i] = d_min
                closest_conflict_agent[i] = j

    # 2) Leader / follower rule -- identical to the range-rate function.
    output = {}
    for i in agent_ids:
        j = closest_conflict_agent[i]
        leader_id = None
        leader_ttc = np.inf
        leader_d_min = np.inf
        if j is not None and min_ttc[i] > min_ttc[j]:
            leader_id = j
            leader_ttc = min_ttc[j]
            leader_d_min = min_d_min[j]
        output[i] = {
            "leader_id": leader_id,
            "leader_ttc": float(leader_ttc),
            "ego_ttc_to_conflict": float(min_ttc[i]),
            "leader_d_min": float(leader_d_min),
        }
    return output


# -------- small self-test (run as a script) ---------------------------------
if __name__ == "__main__":  # pragma: no cover

    def _state(x, y, vx, vy):
        return {
            "position": np.array([x, y], dtype=np.float32),
            "velocity": np.array([vx, vy], dtype=np.float32),
        }

    # Case 1: head-on along x, 10 m apart, each 5 m/s -> meet at midpoint in 1 s,
    # d_min = 0, so CPA TTC = 1.0 s for both.
    states = {"A": _state(0.0, 0.0, 5.0, 0.0), "B": _state(10.0, 0.0, -5.0, 0.0)}
    out = compute_pairwise_cpa_leaders(states, neighbor_radius=20.0, r_safe=3.0)
    print("head-on   :", out)

    # Case 2: perpendicular crossing, 10 m apart at right angle approach,
    # both 5 m/s aimed at origin. They are co-incident at the origin in 2 s
    # respectively (10/5), so CPA should fire and d_min ~ 0.
    states = {"A": _state(-10.0, 0.0, 5.0, 0.0), "B": _state(0.0, -10.0, 0.0, 5.0)}
    out = compute_pairwise_cpa_leaders(states, neighbor_radius=20.0, r_safe=3.0)
    print("perp.cross:", out)

    # Case 3: same lane, B behind A, both moving +x. A=5 m/s, B=7 m/s
    # -> B is closing on A from behind, finite TTC, d_min = 0.
    states = {"A": _state(10.0, 0.0, 5.0, 0.0), "B": _state(0.0, 0.0, 7.0, 0.0)}
    out = compute_pairwise_cpa_leaders(states, neighbor_radius=20.0, r_safe=3.0)
    print("rear-end  :", out)

    # Case 4: side-by-side parallel, same lane shift of 5 m, same velocity.
    # |dv| ~ 0 -> no future contact -> TTC inf (correctly rejected).
    states = {"A": _state(0.0, 0.0, 5.0, 0.0), "B": _state(0.0, 5.0, 5.0, 0.0)}
    out = compute_pairwise_cpa_leaders(states, neighbor_radius=20.0, r_safe=3.0)
    print("parallel  :", out)
