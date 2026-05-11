"""
Stackelberg algorithm (no CoPO coordination underneath).

Inherits from IPPO (PPO + parameter sharing) and adds, on top:
    1. A custom torch model with:
         - decentralized actor pi_theta(a | o_i)            (input: o_i, dim = obs_dim)
         - centralized critic V_phi(o_i, a_hat_leader, leader_present)
                                                            (input: obs_dim + act_dim + 1)
    2. A custom policy that, in postprocess_trajectory(), per timestep:
         - reads info[agent_id]["leader_id"] (set by CCEnv via pairwise TTC),
         - finds the leader's CUR_OBS at the same env timestep from
           other_agent_batches,
         - computes the leader's action signal:
             - "predicted_detached" (default, matches Slide 5):
                   a_hat_leader = pi_theta(o_leader).deterministic_sample().detach()
             - "sampled":
                   a_hat_leader = leader's actually sampled action from rollout,
         - builds CENTRALIZED_CRITIC_OBS = concat([o_i, a_hat_leader, leader_present]),
         - re-computes VF_PREDS via the centralized critic,
         - re-computes GAE / value targets with this VF_PREDS.
    3. A custom loss that uses model.central_value_function(...) instead of
       the default value head.

Crucially, this file does NOT touch:
    - LCF parameters / LCF outer-loop optimizer
    - LCF-weighted ("coordinated") reward
    - neighbour-reward fusion or global value head
    - communication channels
None of those concepts exist here. Stackelberg only enters through the
critic's input.
"""

import gym
import numpy as np
from gym.spaces import Box

from ray.rllib.evaluation.postprocessing import (
    Postprocessing,
    compute_advantages,
)
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.torch.misc import (
    AppendBiasLayer,
    SlimFC,
    normc_initializer,
)
from ray.rllib.models.torch.torch_action_dist import TorchDiagGaussian
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import (
    convert_to_torch_tensor,
    explained_variance,
    sequence_mask,
    warn_if_infinite_kl_divergence,
)
from ray.rllib.utils.typing import TrainerConfigDict

from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy

from copo.torch_copo.algo_ippo import IPPOConfig, IPPOTrainer

torch, nn = try_import_torch()

# Field names used in the SampleBatch. Kept locally on purpose so the
# Stackelberg stack does not depend on CoPO/CCPPO module-level constants.
CENTRALIZED_CRITIC_OBS = "centralized_critic_obs"
LEADER_ACTION = "leader_action"
LEADER_PRESENT = "leader_present"


# =======================================================================
# Model
# =======================================================================
class StackelbergModel(TorchModelV2, nn.Module):
    """
    Two-headed model:
        Actor  : MLP over ego obs only -> action distribution params.
        Critic : MLP over [obs ; a_hat_leader ; leader_present] -> scalar V.

    The actor architecture intentionally mirrors the IPPO baseline: the
    only thing Stackelberg changes is the critic input shape.
    """

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config,
        name: str,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        hiddens = list(model_config.get("fcnet_hiddens", [])) + list(
            model_config.get("post_fcnet_hiddens", [])
        )
        activation = model_config.get("fcnet_activation")
        if not model_config.get("fcnet_hiddens", []):
            activation = model_config.get("post_fcnet_activation")
        self.free_log_std = model_config.get("free_log_std", False)

        if self.free_log_std:
            assert num_outputs % 2 == 0, (
                "num_outputs must be divisible by two when free_log_std=True",
                num_outputs,
            )
            num_outputs = num_outputs // 2

        obs_dim = int(np.product(obs_space.shape))
        assert hasattr(action_space, "shape") and len(action_space.shape) == 1, (
            "Stackelberg model expects a flat continuous Box action space; got "
            "{}".format(action_space)
        )
        act_dim = int(np.product(action_space.shape))

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        # Leader signal = leader action (act_dim) + binary "leader present" flag (1).
        self.leader_signal_dim = act_dim + 1
        self.critic_input_dim = obs_dim + self.leader_signal_dim

        # ----- Actor (decentralized: ego obs only) -----
        actor_layers = []
        prev = obs_dim
        for size in hiddens[:-1] if hiddens else []:
            actor_layers.append(
                SlimFC(
                    in_size=prev,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation,
                )
            )
            prev = size
        if hiddens:
            actor_layers.append(
                SlimFC(
                    in_size=prev,
                    out_size=hiddens[-1],
                    initializer=normc_initializer(1.0),
                    activation_fn=activation,
                )
            )
            prev = hiddens[-1]
        self._hidden_layers = nn.Sequential(*actor_layers)

        self._logits = SlimFC(
            in_size=prev,
            out_size=num_outputs,
            initializer=normc_initializer(0.01),
            activation_fn=None,
        )
        if self.free_log_std and self._logits is not None:
            self._append_free_log_std = AppendBiasLayer(num_outputs)

        # ----- Centralized critic (obs + leader signal) -----
        critic_layers = []
        prev_vf = self.critic_input_dim
        for size in hiddens:
            critic_layers.append(
                SlimFC(
                    in_size=prev_vf,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation,
                )
            )
            prev_vf = size
        self._value_branch_separate = nn.Sequential(*critic_layers)
        self._value_branch = SlimFC(
            in_size=prev_vf,
            out_size=1,
            initializer=normc_initializer(0.01),
            activation_fn=None,
        )

        # ----- View requirements so RLlib carries these fields into train batches -----
        self.view_requirements[CENTRALIZED_CRITIC_OBS] = ViewRequirement(
            space=Box(low=-np.inf, high=np.inf, shape=(self.critic_input_dim,))
        )
        self.view_requirements[LEADER_ACTION] = ViewRequirement(
            space=Box(low=-np.inf, high=np.inf, shape=(act_dim,))
        )
        self.view_requirements[LEADER_PRESENT] = ViewRequirement(
            space=Box(low=0.0, high=1.0, shape=(1,))
        )
        # Required so that other_agent_batches[*][SampleBatch.ACTIONS] is available
        # at postprocess time (used by the "sampled" leader-action source).
        self.view_requirements[SampleBatch.ACTIONS] = ViewRequirement(
            space=action_space
        )

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"].float()
        obs = obs.reshape(obs.shape[0], -1)
        features = self._hidden_layers(obs)
        logits = self._logits(features)
        if self.free_log_std:
            logits = self._append_free_log_std(logits)
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        raise ValueError(
            "StackelbergModel uses central_value_function(cobs); "
            "do not call value_function() directly."
        )

    def central_value_function(self, cobs):
        """cobs has shape (B, obs_dim + act_dim + 1)."""
        if not isinstance(cobs, torch.Tensor):
            cobs = convert_to_torch_tensor(cobs)
        cobs = cobs.float()
        x = self._value_branch_separate(cobs)
        return torch.reshape(self._value_branch(x), [-1])


ModelCatalog.register_custom_model("stackelberg_model", StackelbergModel)


# =======================================================================
# Policy
# =======================================================================
class StackelbergPolicy(PPOTorchPolicy):
    """
    PPO policy with a centralized critic conditioned on the TTC-selected leader.

    The actor side (sampling, action distribution, action logp) is unchanged
    from PPO/IPPO. Only postprocess_trajectory() and the value-function loss
    differ.
    """

    def extra_action_out(self, input_dict, state_batches, model, action_dist):
        return {}

    # ---------------- helpers ----------------
    def _resolve_dist_class(self):
        """Prefer the policy's own dist_class; fall back to TorchDiagGaussian."""
        if hasattr(self, "dist_class") and self.dist_class is not None:
            return self.dist_class
        return TorchDiagGaussian

    def _predict_leader_action(self, leader_obs_arr, leader_present_arr):
        """
        leader_obs_arr     : (T, obs_dim) np.float32. Zero where no leader.
        leader_present_arr : (T, 1)       np.float32. 1.0 where a leader exists.

        Returns (T, act_dim) np.float32 of pi_theta(o_leader).deterministic_sample(),
        with rows zeroed out where no leader is present.
        """
        if leader_obs_arr.shape[0] == 0:
            return np.zeros((0, self.action_space.shape[0]), dtype=np.float32)

        obs_t = convert_to_torch_tensor(leader_obs_arr.astype(np.float32), self.device)
        # Forward through the shared actor; CCModel-style forward expects "obs_flat".
        input_dict = {SampleBatch.OBS: obs_t, "obs_flat": obs_t}
        logits, _ = self.model(input_dict, [], None)
        dist_cls = self._resolve_dist_class()
        dist = dist_cls(logits, self.model)
        # Deterministic sample = mean for diagonal Gaussian. Detached by no_grad.
        pred = dist.deterministic_sample().detach()
        pred_np = pred.cpu().numpy().astype(np.float32)
        return pred_np * leader_present_arr  # zero out non-leader rows

    def _leader_arrays_from_rollout(self, sample_batch, other_agent_batches, T, obs_dim, act_dim):
        """
        Real rollout only: align ego rows with the TTC leader's trajectory at the same env time `t`.
        Returns leader_obs_arr, sampled_leader_action_arr, leader_present_arr (last is (T,1) float32).
        """
        leader_obs_arr = np.zeros((T, obs_dim), dtype=np.float32)
        sampled_leader_action_arr = np.zeros((T, act_dim), dtype=np.float32)
        leader_present_arr = np.zeros((T, 1), dtype=np.float32)

        infos = sample_batch[SampleBatch.INFOS]
        t_arr = sample_batch["t"]

        for index in range(T):
            info = infos[index]
            if not isinstance(info, dict):
                continue
            leader_id = info.get("leader_id", None)
            if leader_id is None:
                continue
            if not other_agent_batches or leader_id not in other_agent_batches:
                continue

            _, leader_batch = other_agent_batches[leader_id]
            t_env = t_arr[index]
            same_t = np.where(leader_batch["t"] == t_env)[0]
            if len(same_t) == 0:
                continue

            idx = int(same_t[0])
            leader_obs_arr[index] = leader_batch[SampleBatch.CUR_OBS][idx].astype(np.float32)
            sampled_leader_action_arr[index] = leader_batch[SampleBatch.ACTIONS][idx].astype(np.float32)
            leader_present_arr[index, 0] = 1.0

        return leader_obs_arr, sampled_leader_action_arr, leader_present_arr

    # ---------------- main hook ----------------
    def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
        """
        RLlib calls this in two modes:

        1) Policy construction (`episode is None`): a tiny synthetic batch so the policy can
           infer shapes and run `loss()` once. There is no multi-agent episode object and no
           `other_agent_batches` — leader fields stay zero. We still must run `compute_advantages`
           so `Postprocessing.ADVANTAGES` / `VALUE_TARGETS` exist; otherwise the init-time
           `loss()` call raises KeyError (see Ray RLlib PPOTorchPolicy._initialize_loss_from_dummy_batch).

        2) Normal training (`episode` is set): fill leader signal from `infos` + other agents'
           batches, recompute VF_PREDS from the centralized critic, then GAE on top.
        """
        with torch.no_grad():
            obs = sample_batch[SampleBatch.CUR_OBS]
            T = obs.shape[0]
            obs_dim = obs.shape[1]
            act_dim = int(np.product(self.action_space.shape))
            critic_input_dim = obs_dim + act_dim + 1  # + leader_present flag

            policy_init_dummy_batch = episode is None

            cobs = np.zeros((T, critic_input_dim), dtype=np.float32)
            cobs[:, :obs_dim] = obs.astype(np.float32)
            leader_action_arr = np.zeros((T, act_dim), dtype=np.float32)
            leader_present_arr = np.zeros((T, 1), dtype=np.float32)

            if policy_init_dummy_batch:
                # Record critic width once; leader slots remain zero (no cross-agent data).
                self.centralized_critic_obs_dim = critic_input_dim
            else:
                leader_obs_arr, sampled_leader_action_arr, leader_present_arr = (
                    self._leader_arrays_from_rollout(
                        sample_batch, other_agent_batches, T, obs_dim, act_dim
                    )
                )
                source = self.config.get("leader_action_source", "predicted_detached")
                if source == "predicted_detached":
                    leader_action_arr = self._predict_leader_action(
                        leader_obs_arr, leader_present_arr
                    )
                elif source == "sampled":
                    leader_action_arr = sampled_leader_action_arr * leader_present_arr
                else:
                    raise ValueError(
                        "Unknown leader_action_source: {}".format(source)
                    )

                cobs[:, obs_dim:obs_dim + act_dim] = leader_action_arr
                cobs[:, obs_dim + act_dim:obs_dim + act_dim + 1] = leader_present_arr

            sample_batch[LEADER_ACTION] = leader_action_arr.astype(np.float32)
            sample_batch[LEADER_PRESENT] = leader_present_arr.astype(np.float32)
            sample_batch[CENTRALIZED_CRITIC_OBS] = cobs

            cobs_t = convert_to_torch_tensor(cobs, self.device)
            sample_batch[SampleBatch.VF_PREDS] = (
                self.model.central_value_function(cobs_t)
                .cpu()
                .detach()
                .numpy()
                .astype(np.float32)
            )

            if sample_batch[SampleBatch.DONES][-1]:
                last_r = 0.0
            else:
                last_r = sample_batch[SampleBatch.VF_PREDS][-1]

            sample_batch = compute_advantages(
                sample_batch,
                last_r,
                self.config["gamma"],
                self.config["lambda"],
                use_gae=self.config["use_gae"],
                use_critic=self.config.get("use_critic", True),
            )
        return sample_batch

    # ---------------- loss (PPO with centralized critic) ----------------
    def loss(self, model, dist_class, train_batch):
        """PPO loss, but the value head is `model.central_value_function(cobs)`."""
        logits, state = model(train_batch)
        curr_action_dist = dist_class(logits, model)

        if state:
            B = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = logits.shape[0] // B
            mask = sequence_mask(
                train_batch[SampleBatch.SEQ_LENS],
                max_seq_len,
                time_major=model.is_time_major(),
            )
            mask = torch.reshape(mask, [-1])
            num_valid = torch.sum(mask)

            def reduce_mean_valid(t):
                return torch.sum(t[mask]) / num_valid
        else:
            mask = None
            reduce_mean_valid = torch.mean

        prev_action_dist = dist_class(
            train_batch[SampleBatch.ACTION_DIST_INPUTS], model
        )

        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
            - train_batch[SampleBatch.ACTION_LOGP]
        )

        if self.config["kl_coeff"] > 0.0:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = reduce_mean_valid(action_kl)
            warn_if_infinite_kl_divergence(self, mean_kl_loss)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = reduce_mean_valid(curr_entropy)

        surrogate_loss = torch.min(
            train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
            train_batch[Postprocessing.ADVANTAGES]
            * torch.clamp(
                logp_ratio,
                1 - self.config["clip_param"],
                1 + self.config["clip_param"],
            ),
        )

        assert self.config["use_critic"]
        # ====== Stackelberg-specific value head ======
        value_fn_out = model.central_value_function(
            train_batch[CENTRALIZED_CRITIC_OBS]
        )
        # =============================================

        if self.config["old_value_loss"]:
            current_vf = value_fn_out
            prev_vf = train_batch[SampleBatch.VF_PREDS]
            vf_loss1 = torch.pow(
                current_vf - train_batch[Postprocessing.VALUE_TARGETS], 2.0
            )
            vf_clipped = prev_vf + torch.clamp(
                current_vf - prev_vf,
                -self.config["vf_clip_param"],
                self.config["vf_clip_param"],
            )
            vf_loss2 = torch.pow(
                vf_clipped - train_batch[Postprocessing.VALUE_TARGETS], 2.0
            )
            vf_loss_clipped = torch.max(vf_loss1, vf_loss2)
        else:
            vf_loss = torch.pow(
                value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
            )
            vf_loss_clipped = torch.clamp(vf_loss, 0, self.config["vf_clip_param"])
        mean_vf_loss = reduce_mean_valid(vf_loss_clipped)

        total_loss = reduce_mean_valid(
            -surrogate_loss
            + self.config["vf_loss_coeff"] * vf_loss_clipped
            - self.entropy_coeff * curr_entropy
        )

        if self.config["kl_coeff"] > 0.0:
            total_loss += self.kl_coeff * mean_kl_loss

        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_policy_loss"] = reduce_mean_valid(-surrogate_loss)
        model.tower_stats["mean_vf_loss"] = mean_vf_loss
        model.tower_stats["vf_explained_var"] = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
        )
        model.tower_stats["mean_entropy"] = mean_entropy
        model.tower_stats["mean_kl_loss"] = mean_kl_loss

        return total_loss


# =======================================================================
# Config / Trainer
# =======================================================================
class StackelbergConfig(IPPOConfig):
    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or StackelbergTrainer)

        # Source of the leader action signal that conditions the critic:
        #   "predicted_detached": a_hat_leader = pi_theta(o_leader).mean, .detach()
        #                          (matches Slide 5; default)
        #   "sampled":            actual sampled leader action from the rollout
        self.leader_action_source = "predicted_detached"

        # Bind our custom torch model.
        self.update_from_dict(
            {"model": {"custom_model": "stackelberg_model"}}
        )

    def validate(self):
        super().validate()
        assert self["leader_action_source"] in (
            "predicted_detached",
            "sampled",
        ), "leader_action_source must be 'predicted_detached' or 'sampled'"


class StackelbergTrainer(IPPOTrainer):
    @classmethod
    def get_default_config(cls):
        return StackelbergConfig()

    def get_default_policy_class(self, config: TrainerConfigDict):
        assert config["framework"] == "torch"
        return StackelbergPolicy
