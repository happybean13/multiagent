"""
Stackelberg environment wrapper.

Composition (intentionally minimal):

    MetaDrive env
        + CCEnv mixin   <-- only used for: distance map + pairwise TTC leader
                            info (these are already implemented in CCEnv.step)
        + RLlib MultiAgentEnv adapter (registers a unique env name)

Notably, this wrapper does NOT use:
    - LCFEnv            (no LCF parameter, no LCF-weighted reward)
    - Communication     (comm_method forced to "none")
    - Neighbour reward / global reward shaping

So the per-agent reward returned by the env IS the MetaDrive native reward.
Coordination only enters the algorithm through the centralized critic
input (see algo_stackelberg.py), not through the reward signal.
"""

from copo.torch_copo.utils.env_wrappers import (
    CCEnv,
    get_rllib_compatible_env,
)


def get_stackelberg_env(env_class):
    """
    Compose a Stackelberg-flavoured RLlib MultiAgentEnv from a MetaDrive env.

    Steps:
      1) Mix `CCEnv` into `env_class`. CCEnv adds:
           - a per-step distance map between agents,
           - `info[agent_id]["leader_id" / "leader_ttc" / "ego_ttc_to_conflict"]`
             via pairwise TTC (this is exactly the leader signal Stackelberg uses).
         CCEnv does NOT modify the reward.
      2) Force comm_method="none" so no inter-agent communication is added.
      3) Register the resulting class with RLlib under a unique name
         (`Stackelberg<EnvName>`) so it cannot collide with any LCF/CoPO
         registration of the same underlying env.
    """
    base_name = env_class.__name__

    class _StackelbergEnv(CCEnv, env_class):
        @classmethod
        def default_config(cls):
            cfg = super(_StackelbergEnv, cls).default_config()
            # Make sure the comm channel is OFF: we want plain ego obs.
            cfg["communication"]["comm_method"] = "none"
            return cfg

    name = "Stackelberg{}".format(base_name)
    _StackelbergEnv.__name__ = name
    _StackelbergEnv.__qualname__ = name

    # Registers the class with RLlib's tune registry under `name`,
    # and returns the registered name string (which is what RLlib expects in config["env"]).
    return get_rllib_compatible_env(_StackelbergEnv)
