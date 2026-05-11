"""
Publishable-ablation training: Minimal Stackelberg (no CoPO coordination underneath).

Budget (matches "minimum claim" standard for MetaDrive MARL):
    - 1,000,000 environment steps per seed
    - 2 seeds by default (override with --num-seeds; --start-seed lets you add more later)
    - MultiAgentIntersectionEnv only
    - CPU-only (current PyTorch wheel does not support this machine's RTX 5070 sm_120)

Logging:
    - MultiAgentDrivingCallbacks fills success / crash / out / max_step / length in progress.csv
    - Checkpoints every 10 iters (~ every 20k env steps), keep latest 5
    - Result dir: ./<exp_name>/StackelbergTrainer_StackelbergMultiAgentIntersectionEnv_<run_id>_seed=<s>_<ts>/

To add more seeds later without retraining the existing two:
    PYTHONPATH=$PWD/CoPO/copo_code python CoPO/copo_code/copo/torch_stackelberg/train_minimal_stackelberg_1M.py \
        --exp-name minimal_stackelberg_1M --num-seeds 1 --start-seed 200
"""

from metadrive.envs.marl_envs import MultiAgentIntersectionEnv

from copo.torch_copo.utils.callbacks import MultiAgentDrivingCallbacks
from copo.torch_copo.utils.train import train
from copo.torch_copo.utils.utils import get_train_parser

from copo.torch_stackelberg.algo_stackelberg import StackelbergTrainer
from copo.torch_stackelberg.env_stackelberg import get_stackelberg_env


if __name__ == "__main__":
    parser = get_train_parser()
    parser.add_argument("--start-seed", type=int, default=0)
    args = parser.parse_args()

    exp_name = args.exp_name or "minimal_stackelberg_1M"
    num_seeds = args.num_seeds if args.num_seeds is not None else 2

    stop = {"timesteps_total": 1_000_000}

    config = dict(
        env=get_stackelberg_env(MultiAgentIntersectionEnv),
        env_config=dict(),

        # CPU-only (see top-of-file note about RTX 5070 sm_120 incompatibility).
        num_gpus=0,

        # Stackelberg-specific: use predicted leader action (matches Slide 5).
        leader_action_source="predicted_detached",

        # Same as IPPO baseline column for fair comparison.
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
        num_seeds=num_seeds,
        start_seed=args.start_seed,
        test_mode=args.test,
        custom_callback=MultiAgentDrivingCallbacks,
    )
