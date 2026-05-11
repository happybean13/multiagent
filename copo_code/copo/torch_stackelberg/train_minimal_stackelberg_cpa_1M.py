"""
Training script: Minimal Stackelberg with 2D CPA leader selection (no CoPO).

This is the CPA counterpart of `train_minimal_stackelberg_1M.py`. The only
difference is the env factory: `get_stackelberg_cpa_env` instead of
`get_stackelberg_env`. Everything else (algo, budget, callbacks, vf_clip)
is matched so the two runs are directly comparable.

Budget per seed:
    - 1,000,000 environment steps
    - 3 seeds by default (seeds = {0, 100, 200} via i*100 + start_seed)
    - MultiAgentIntersectionEnv only
    - CPU-only

CPA knobs (forwarded to env_config -> CPACCEnv.default_config):
    cpa_r_safe = 3.0
    cpa_neighbor_radius = 10.0   (kept identical to the range-rate baseline)

Run:
    PYTHONPATH=$PWD/CoPO/copo_code python \
        CoPO/copo_code/copo/torch_stackelberg/train_minimal_stackelberg_cpa_1M.py \
        --exp-name minimal_stackelberg_cpa_1M --num-seeds 3 --start-seed 0
"""

from metadrive.envs.marl_envs import MultiAgentIntersectionEnv

from copo.torch_copo.utils.callbacks import MultiAgentDrivingCallbacks
from copo.torch_copo.utils.train import train
from copo.torch_copo.utils.utils import get_train_parser

from copo.torch_stackelberg.algo_stackelberg import StackelbergTrainer
from copo.torch_stackelberg.env_stackelberg_cpa import get_stackelberg_cpa_env


if __name__ == "__main__":
    parser = get_train_parser()
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--cpa-r-safe", type=float, default=3.0)
    parser.add_argument("--cpa-neighbor-radius", type=float, default=10.0)
    args = parser.parse_args()

    exp_name = args.exp_name or "minimal_stackelberg_cpa_1M"
    num_seeds = args.num_seeds if args.num_seeds is not None else 3

    stop = {"timesteps_total": 1_000_000}

    config = dict(
        env=get_stackelberg_cpa_env(MultiAgentIntersectionEnv),
        env_config=dict(
            cpa_r_safe=args.cpa_r_safe,
            cpa_neighbor_radius=args.cpa_neighbor_radius,
        ),

        num_gpus=0,

        leader_action_source="predicted_detached",

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
