"""
Decomposed SAC — corrected full implementation.

Architecture
------------
  [s, a] → shared MLP encoder → Q_aug_head   (2 heads, clipped double-Q)
                               → Q_sparse_head (2 heads, clipped double-Q)

Reward decomposition
--------------------
  R       = R_dense + R_sparse
  G_soft  = G_aug + G_sparse
           = Σ γ^i (R_dense,i + α H(π(·|s_i)))  +  γ^k R_sparse

Q decomposition
---------------
  Q_soft(s,a) = Q_aug(s,a) + Q_sparse(s,a)

Bellman targets
---------------
  y_aug    = R_aug + γ(1-d)[min Q_aug(s',ã') - α log π(ã'|s')]
  y_sparse = R_sparse + γ(1-d) min Q_sparse(s',ã')
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Union, cast

import gymnasium.spaces as spaces
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.torch_layers import CombinedExtractor
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.sac.policies import SACPolicy


# ─── 1. Replay buffer ────────────────────────────────────────────────────────

class DecomposedReplayBuffer(DictReplayBuffer):
    """
    DictReplayBuffer (required for Dict obs spaces) extended with a
    sparse_rewards array aligned to the main buffer so the training loop
    can retrieve sparse rewards for exactly the same transitions sampled.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sparse_rewards: np.ndarray = np.zeros(
            (self.buffer_size, self.n_envs, 1), dtype=np.float32
        )
        self._last_sample_inds = None
        self._last_env_inds    = None

    def sample(self, batch_size: int, env=None) -> ReplayBufferSamples:
        upper      = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper, size=batch_size)
        self._last_sample_inds = batch_inds
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env=None) -> ReplayBufferSamples:
        # DictReplayBuffer._get_samples calls np.random.randint for env_indices as
        # its first operation.  We snapshot + restore the RNG state so we can see
        # exactly which env_indices the parent will use, then let it run normally.
        rng_state = np.random.get_state()
        self._last_env_inds = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        np.random.set_state(rng_state)
        return super()._get_samples(batch_inds, env=env)

    def get_sparse_rewards(self, device: torch.device) -> torch.Tensor:
        assert self._last_sample_inds is not None, "Call sample() before get_sparse_rewards()."
        raw = self.sparse_rewards[self._last_sample_inds, self._last_env_inds]
        return torch.tensor(raw, device=device, dtype=torch.float32)


# ─── 2. Decomposed critic ────────────────────────────────────────────────────

class DecomposedCritic(ContinuousCritic):
    """
    ContinuousCritic with an additional pair of Q-heads (qf0_sparse, qf1_sparse)
    trained exclusively on the sparse reward stream.

    forward_decomposed preprocesses dict obs before feature extraction, so it
    works correctly whether called from the training loop (already-preprocessed
    DictReplayBuffer tensors) or from the probe sweep (raw dict tensors).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # deepcopy gives identical architecture with fresh independent weights,
        # robust across SB3 versions (no reliance on net_arch/features_dim attrs).
        self.qf0_sparse = copy.deepcopy(self.qf0)
        self.qf1_sparse = copy.deepcopy(self.qf1)

    def _get_qf_input(self, obs: Union[torch.Tensor, dict], actions: torch.Tensor) -> torch.Tensor:
        """Preprocess obs if needed, extract features, concatenate actions."""
        if isinstance(obs, dict):
            obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        features = self.extract_features(obs, self.features_extractor)
        return torch.cat([features, actions], dim=1)

    def forward_decomposed(
        self,
        obs: Union[torch.Tensor, dict],
        actions: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        qf_input = self._get_qf_input(obs, actions)
        return (
            (self.qf0(qf_input), self.qf1(qf_input)),
            (self.qf0_sparse(qf_input), self.qf1_sparse(qf_input)),
        )

    def forward(
        self,
        obs: Union[torch.Tensor, dict],
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard forward — keeps SB3 internals working unchanged."""
        (q0, q1), _ = self.forward_decomposed(obs, actions)
        return q0, q1


# ─── 3. Policy ───────────────────────────────────────────────────────────────

class DecomposedSACPolicy(SACPolicy):
    def __init__(self, observation_space, *args, **kwargs):
        if isinstance(observation_space, spaces.Dict):
            kwargs.setdefault("features_extractor_class", CombinedExtractor)
        super().__init__(observation_space, *args, **kwargs)

    def make_critic(self, features_extractor=None) -> DecomposedCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return DecomposedCritic(**critic_kwargs).to(self.device)


# ─── 4. SAC subclass ─────────────────────────────────────────────────────────

class DecomposedSAC(SAC):
    """
    SAC with a two-head critic trained on decomposed reward streams.

    Environment contract: each step() info dict must contain:
        info["reward_dense"]  : float  — dense reward for this step
        info["reward_sparse"] : float  — non-zero only at terminal steps
    """

    policy_aliases = {
        **SAC.policy_aliases,
        "MlpPolicy":        DecomposedSACPolicy,
        "MultiInputPolicy": DecomposedSACPolicy,
    }

    def _setup_model(self) -> None:
        super()._setup_model()
        self.replay_buffer = DecomposedReplayBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            device=self.device,
            n_envs=self.n_envs,
            **self.replay_buffer_kwargs,
        )

    def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, dones, infos) -> None:
        reward_dense = np.array(
            [info.get("reward_dense", r) for info, r in zip(infos, reward)],
            dtype=np.float32,
        )
        reward_sparse = np.array(
            [info.get("reward_sparse", 0.0) for info in infos],
            dtype=np.float32,
        )
        super()._store_transition(replay_buffer, buffer_action, new_obs, reward_dense, dones, infos)
        pos = (replay_buffer.pos - 1) % replay_buffer.buffer_size
        replay_buffer.sparse_rewards[pos] = reward_sparse.reshape(-1, 1)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        for _ in range(gradient_steps):
            self._n_updates += 1

            replay_data    = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            sparse_rewards = self.replay_buffer.get_sparse_rewards(self.device)

            # Mirrors SB3 SAC pattern: self.ent_coef stays "auto" (string) when
            # entropy is learned, so always derive a proper tensor for arithmetic.
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = torch.exp(self.log_ent_coef.detach())
            else:
                ent_coef = self.ent_coef_tensor

            # cast: SB3 types critic/critic_target as ContinuousCritic; at runtime
            # make_critic() returns DecomposedCritic, so forward_decomposed exists.
            critic        = cast(DecomposedCritic, self.critic)
            critic_target = cast(DecomposedCritic, self.critic_target)

            with torch.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                (next_q_aug_0, next_q_aug_1), (next_q_sp_0, next_q_sp_1) = \
                    critic_target.forward_decomposed(replay_data.next_observations, next_actions)

                next_q_aug = torch.min(next_q_aug_0, next_q_aug_1)
                next_q_sp  = torch.min(next_q_sp_0,  next_q_sp_1)

                target_aug = (
                    replay_data.rewards
                    + (1.0 - replay_data.dones) * self.gamma
                    * (next_q_aug - ent_coef * next_log_prob)
                )
                target_sparse = sparse_rewards + (1.0 - replay_data.dones) * self.gamma * next_q_sp

            (q_aug_0, q_aug_1), (q_sp_0, q_sp_1) = \
                critic.forward_decomposed(replay_data.observations, replay_data.actions)

            loss_aug    = 0.5 * (nn.functional.mse_loss(q_aug_0, target_aug)
                                 + nn.functional.mse_loss(q_aug_1, target_aug))
            loss_sparse = 0.5 * (nn.functional.mse_loss(q_sp_0, target_sparse)
                                 + nn.functional.mse_loss(q_sp_1, target_sparse))
            critic_loss = loss_aug + loss_sparse

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            (q_aug_pi_0, q_aug_pi_1), (q_sp_pi_0, q_sp_pi_1) = \
                critic.forward_decomposed(replay_data.observations, actions_pi)

            min_q_pi   = torch.min(q_aug_pi_0, q_aug_pi_1) + torch.min(q_sp_pi_0, q_sp_pi_1)
            actor_loss = (ent_coef * log_prob - min_q_pi).mean()

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if self.ent_coef_optimizer is not None:
                with torch.no_grad():
                    _, log_prob = self.actor.action_log_prob(replay_data.observations)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy)).mean()
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
                self.ent_coef = torch.exp(self.log_ent_coef.detach())

            if self._n_updates % self.target_update_interval == 0:
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)


# ─── 5. Gradient extraction for dominance analysis ───────────────────────────

def compute_exact_gradients(
    model: SAC,
    obs:   Union[np.ndarray, dict],
    action: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes dQ_aug/da and dQ_sparse/da via backprop.

    obs    : flat array (MlpPolicy) or dict of arrays (MultiInputPolicy), shape (B, ...)
    action : (B, Act_Dim)
    returns: grad_aug, grad_sparse, q_aug_vals, q_sparse_vals
    """
    model.policy.set_training_mode(False)

    if isinstance(obs, dict):
        obs_t = {k: torch.as_tensor(v, dtype=torch.float32, device=model.device) for k, v in obs.items()}
    else:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=model.device)

    act_t = torch.as_tensor(action, dtype=torch.float32, device=model.device).requires_grad_(True)

    (q_aug_0, q_aug_1), (q_sp_0, q_sp_1) = cast(DecomposedCritic, model.critic).forward_decomposed(obs_t, act_t)
    q_aug = torch.min(q_aug_0, q_aug_1)
    q_sp  = torch.min(q_sp_0,  q_sp_1)

    grad_aug = torch.autograd.grad(
        q_aug, act_t, grad_outputs=torch.ones_like(q_aug), retain_graph=True
    )[0].detach().cpu().numpy()

    grad_sparse = torch.autograd.grad(
        q_sp, act_t, grad_outputs=torch.ones_like(q_sp)
    )[0].detach().cpu().numpy()

    return (
        grad_aug,
        grad_sparse,
        q_aug.detach().cpu().numpy().flatten(),
        q_sp.detach().cpu().numpy().flatten(),
    )


def dominance_metrics(
    grad_aug:    np.ndarray,
    grad_sparse: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (lambda, cos_theta) for a batch of gradient pairs. Both shape (B,)."""
    norm_aug = np.maximum(np.linalg.norm(grad_aug,    axis=1, keepdims=True), 1e-9)
    norm_sp  = np.maximum(np.linalg.norm(grad_sparse, axis=1, keepdims=True), 1e-9)
    lam       = (norm_sp / norm_aug).flatten()
    dot       = np.sum(grad_aug * grad_sparse, axis=1, keepdims=True)
    cos_theta = (dot / (norm_aug * norm_sp)).flatten()
    return lam, cos_theta