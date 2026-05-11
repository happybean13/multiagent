"""
torch_stackelberg
=================

Minimal Stackelberg multi-agent setup, deliberately built WITHOUT CoPO's
coordination stack underneath. CoPO is treated only as a separate baseline.

What is INHERITED from the IPPO baseline (matches Slide 5: "Inherit from baseline"):
  - PPO + parameter sharing (one shared actor across agents)
  - Actor architecture pi_theta(a | o_i) over ego observation only
  - MetaDrive MultiAgentIntersectionEnv with its native per-agent reward
    (no LCF, no neighbour reward, no global reward shaping)

What is ADDED on top (matches Slide 5: "Stackelberg additions"):
  - TTC-based leader selection: at each step, every agent picks at most one
    "leader" using pairwise time-to-collision (env-side, in info[]).
  - Centralized critic V_phi(o_i, a_hat_leader, leader_present), where
    a_hat_leader = pi_theta(o_leader).detach() (or the actually sampled
    leader action, behind a config flag).

What is REMOVED versus the previous Stackelberg-on-top-of-CoPO fork:
  - LCF parameters and the LCF outer-loop optimizer
  - LCF-weighted ("coordinated") reward
  - Neighbour reward fusion / global value head
  - Inter-agent communication channel
"""
