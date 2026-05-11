"""
Publishable-ablation training: CoPO BASELINE (the original fork, with LCF coordination).

This is the "CoPO column" of the ablation table. It is intentionally an unmodified
CoPO setup so the comparison vs Minimal Stackelberg is honest. We do NOT touch any
Stackelberg code here -- this script just re-runs the upstream CoPOTrainer with the
same env, the same logger, and the same per-seed budget as the Stackelberg run.

Budget (matches the Stackelberg-only ablation):
    - 1,000,000 environment steps per seed
    - 2 seeds by default (override with --num-seeds; --start-seed for adding more later)
    - MultiAgentIntersectionEnv (wrapped with LCFEnv via get_lcf_env, as the CoPO paper does)
    - CPU-only

Lives in torch_stackelberg/ on purpose: this file is part of the *ablation harness*,
not part of the CoPO algorithm package -- so we don't overwrite torch_copo/train_copo.py
(which has been previously edited for shorter sanity runs).

To add more seeds later:
    PYTHONPATH=$PWD/CoPO/copo_code python CoPO/copo_code/copo/torch_stackelberg/train_copo_baseline_1M.py \
        --exp-name copo_baseline_1M --num-seeds 1 --start-seed 200
"""

from metadrive.envs.marl_envs import MultiAgentIntersectionEnv

from copo.torch_copo.algo_copo import (
    CoPOTrainer,
    USE_CENTRALIZED_CRITIC,
)
from copo.torch_copo.utils.callbacks import MultiAgentDrivingCallbacks
from copo.torch_copo.utils.env_wrappers import get_lcf_env, get_rllib_compatible_env
from copo.torch_copo.utils.train import train
from copo.torch_copo.utils.utils import get_train_parser


if __name__ == "__main__":
    parser = get_train_parser()
    parser.add_argument("--start-seed", type=int, default=0)
    args = parser.parse_args()

    exp_name = args.exp_name or "copo_baseline_1M"
    num_seeds = args.num_seeds if args.num_seeds is not None else 2

    stop = {"timesteps_total": 1_000_000}

    config = dict(
        # Standard CoPO env: MetaDrive intersection wrapped with LCFEnv (the LCF reward
        # shaping that defines CoPO's "coordination underneath").
        env=get_rllib_compatible_env(get_lcf_env(MultiAgentIntersectionEnv)),
        env_config=dict(neighbours_distance=40),

        # CPU-only (see RTX 5070 sm_120 note in the Stackelberg script).
        num_gpus=0,
        num_cpus_per_worker=0.1,

        # CoPO outer-loop hyperparameters as in the original train_copo.py.
        initial_svo_std=0.1,
        svo_lr=1e-4,
        svo_num_iters=5,
        use_global_value=True,
        **{USE_CENTRALIZED_CRITIC: False},
    )

    train(
        CoPOTrainer,
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
