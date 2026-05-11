"""
Train minimal Stackelberg (no CoPO coordination underneath).

Setup matches Slide 5 ("Stackelberg vs CoPO" – the Stackelberg column):

    Inherited from IPPO baseline:
        - Env       : MetaDrive MultiAgentIntersectionEnv
        - Reward    : MetaDrive native per-agent reward (no LCF, no nei/global)
        - Algorithm : PPO + parameter sharing (single shared actor)
        - Actor arch: pi_theta(a | o_i)  -- ego obs only

    Added by Stackelberg:
        - TTC-based leader selection in env infos (via CCEnv mixin)
        - Centralized critic V_phi(o_i, a_hat_leader, leader_present),
          with a_hat_leader = pi_theta(o_leader).detach() by default.

Run:
    PYTHONPATH=$PWD/copo_code python copo_code/copo/torch_stackelberg/train_minimal_stackelberg.py \
        --exp-name minimal_stackelberg_100k
"""

from metadrive.envs.marl_envs import MultiAgentIntersectionEnv

from copo.torch_copo.utils.callbacks import MultiAgentDrivingCallbacks
from copo.torch_copo.utils.train import train
from copo.torch_copo.utils.utils import get_train_parser

from copo.torch_stackelberg.algo_stackelberg import StackelbergTrainer
from copo.torch_stackelberg.env_stackelberg import get_stackelberg_env


if __name__ == "__main__":
    args = get_train_parser().parse_args()
    exp_name = args.exp_name or "minimal_stackelberg_100k"

    # Match the IPPO/CoPO baseline budget so the comparison column is fair:
    # 100 training iterations ~= 100k env steps with the default IPPOConfig
    # (rollout_fragment_length=200, train_batch_size=2000).
    stop = {"training_iteration": 100}

    config = dict(
        # Stackelberg-flavoured intersection env: native reward + TTC leader info.
        env=get_stackelberg_env(MultiAgentIntersectionEnv),
        env_config=dict(),

        # Resource: align with train_ippo.py / train_copo.py defaults.
        num_gpus=0.25 if args.num_gpus != 0 else 0,

        # Stackelberg-specific knob: which leader action conditions the critic?
        # "predicted_detached" matches Slide 5; "sampled" reproduces the older fork.
        leader_action_source="predicted_detached",

        # Same vf_clip_param as the picked IPPO baseline for the comparison table.
        vf_clip_param=100,
    )

    train(
        StackelbergTrainer,
        exp_name=exp_name,
        keep_checkpoints_num=5,
        checkpoint_freq=10,
        stop=stop,
        config=config,
        num_gpus=args.num_gpus,
        num_seeds=1,
        test_mode=args.test,
        custom_callback=MultiAgentDrivingCallbacks,
    )
