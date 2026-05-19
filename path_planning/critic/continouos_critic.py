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
  y_sparse = R_sparse   (terminal-only; no bootstrap)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple

from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.sac.policies import SACPolicy


# ─── 1. Replay buffer that stores sparse rewards and exposes sample indices ──

class DecomposedReplayBuffer(ReplayBuffer):
    """
    Extends SB3's ReplayBuffer with:
      • a `sparse_rewards` array (same shape as `rewards`)
      • `_last_sample_inds` so the training loop can index sparse_rewards
        with the *same* indices used for the main batch.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sparse_rewards: np.ndarray = np.zeros(
            (self.buffer_size, self.n_envs, 1), dtype=np.float32
        )
        self._last_sample_inds = None
        self._last_env_inds = None

    # ------------------------------------------------------------------
    # Override _sample_indices (SB3 ≥ 1.8 exposes this hook).
    # For older SB3 we override sample() directly.
    # ------------------------------------------------------------------
    def sample(
        self,
        batch_size: int,
        env=None,
    ) -> ReplayBufferSamples:
        """Capture both time and environment indices for vectorisation."""
        upper = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper, size=batch_size)
        
        # Capture environment indices (mimicking SB3 internal logic)
        if env is None:
            env_indices = np.random.randint(0, self.n_envs, size=batch_size)
        else:
            env_indices = np.zeros(batch_size, dtype=np.int64)

        self._last_sample_inds = batch_inds
        self._last_env_inds = env_indices
        
        # Use SB3's internal _get_samples with the specific indices
        return self._get_samples(batch_inds, env_indices, env=env)

    def get_sparse_rewards(self, device: torch.device) -> torch.Tensor:
        """
        Return the sparse rewards that correspond to the last sample() call.
        Shape: (batch_size, 1).
        """
        assert self._last_sample_inds is not None, (
            "Call sample() before get_sparse_rewards()."
        )
        
        # Index into (buffer_size, n_envs, 1) using captured indices
        raw = self.sparse_rewards[self._last_sample_inds, self._last_env_inds]
        return torch.tensor(raw, device=device, dtype=torch.float32)


# ─── 2. Decomposed critic ────────────────────────────────────────────────────

class DecomposedCritic(ContinuousCritic):
    """
    Shares the encoder trunk with the standard ContinuousCritic but adds a
    second pair of Q-heads fitted exclusively to G_sparse returns.

    SB3 builds self.qf0 / self.qf1 (full MLPs) in super().__init__.
    We mirror that structure for the sparse heads by reusing
    `self.net_arch` and `self.activation_fn` via create_mlp.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Build sparse heads with the same depth/width as the aug heads.
        # SB3's ContinuousCritic stores net_arch and activation_fn on self.
        from stable_baselines3.common.torch_layers import create_mlp

        action_dim  = self.action_space.shape[0]
        input_dim   = self.features_dim + action_dim

        def _make_sparse_head() -> nn.Sequential:
            net = create_mlp(
                input_dim,
                1,
                self.net_arch,
                self.activation_fn,
            )
            return nn.Sequential(*net)

        self.qf0_sparse = _make_sparse_head()
        self.qf1_sparse = _make_sparse_head()

    # ------------------------------------------------------------------

    def forward_decomposed(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        """
        Returns
        -------
        (q_aug_0, q_aug_1)       clipped double-Q for augmented signal
        (q_sparse_0, q_sparse_1) clipped double-Q for sparse signal
        """
        features  = self.extract_features(obs, self.features_extractor)
        qf_input  = torch.cat([features, actions], dim=1)

        q_aug_0    = self.qf0(qf_input)
        q_aug_1    = self.qf1(qf_input)
        q_sparse_0 = self.qf0_sparse(qf_input)
        q_sparse_1 = self.qf1_sparse(qf_input)

        return (q_aug_0, q_aug_1), (q_sparse_0, q_sparse_1)

    def forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard forward — keeps SB3 internals working unchanged."""
        (q0, q1), _ = self.forward_decomposed(obs, actions)
        return q0, q1


# ─── 3. Policy that builds the decomposed critic ─────────────────────────────

class DecomposedSACPolicy(SACPolicy):
    """SACPolicy subclass that swaps in DecomposedCritic."""

    def make_critic(self, features_extractor=None) -> DecomposedCritic:
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs, features_extractor
        )
        return DecomposedCritic(**critic_kwargs).to(self.device)


# ─── 4. SAC subclass with decomposed training step ───────────────────────────

class DecomposedSAC(SAC):
    """
    SAC with a two-head critic.

    The replay buffer stores:
        rewards        → R_aug  = R_dense  (dense signal)
        sparse_rewards → R_sparse          (terminal-only signal)

    Usage
    -----
        model = DecomposedSAC("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=500_000)

    Environment contract
    --------------------
    Each `step()` must return an info dict containing:
        info["reward_dense"]  : float   — the dense reward for this step
        info["reward_sparse"] : float   — non-zero only at terminal steps
    """

    # Register the custom policy under the "MlpPolicy" alias.
    policy_aliases = {**SAC.policy_aliases, "MlpPolicy": DecomposedSACPolicy, "MultiInputPolicy": DecomposedSACPolicy}

    def _setup_model(self) -> None:
        super()._setup_model()
        # Replace the buffer SB3 just built with our decomposed version.
        self.replay_buffer = DecomposedReplayBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            device=self.device,
            n_envs=self.n_envs,
            optimize_memory_usage=self.optimize_memory_usage,
        )

    # ------------------------------------------------------------------

    def _store_transition(
        self,
        replay_buffer: DecomposedReplayBuffer,
        buffer_action: np.ndarray,
        new_obs,
        reward: np.ndarray,
        dones: np.ndarray,
        infos,
    ) -> None:
        """
        Extract dense / sparse rewards from info, store the dense reward
        as the standard SB3 `rewards` field, and write the sparse reward
        into our custom `sparse_rewards` array at the same position.
        """
        reward_dense = np.array([
            info.get("reward_dense", r) for info, r in zip(infos, reward)
        ], dtype=np.float32)

        reward_sparse = np.array([
            info.get("reward_sparse", 0.0) for info in infos
        ], dtype=np.float32)

        # Store with the dense reward so SB3's `replay_data.rewards` == R_dense.
        super()._store_transition(
            replay_buffer, buffer_action, new_obs,
            reward_dense, dones, infos,
        )

        # Write sparse reward at the slot that was just filled.
        # replay_buffer.pos was already incremented by super(), so the
        # latest entry is at (pos - 1) % buffer_size.
        pos = (replay_buffer.pos - 1) % replay_buffer.buffer_size
        replay_buffer.sparse_rewards[pos] = reward_sparse.reshape(-1, 1)

    # ------------------------------------------------------------------

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(
            [self.actor.optimizer, self.critic.optimizer]
        )

        for _ in range(gradient_steps):
            self._n_updates += 1

            # ── Sample ────────────────────────────────────────────────
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )
            # Retrieve sparse rewards for the *same* sampled indices.
            sparse_rewards = self.replay_buffer.get_sparse_rewards(
                self.device
            )  # shape (B, 1)

            # ── Bellman targets (no gradient) ─────────────────────────
            with torch.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations
                )
                (next_q_aug_0, next_q_aug_1), (next_q_sp_0, next_q_sp_1) = \
                    self.critic.forward_decomposed(
                        replay_data.next_observations, next_actions
                    )

                # Clipped double-Q next values
                next_q_aug = torch.min(next_q_aug_0, next_q_aug_1)
                next_q_sp  = torch.min(next_q_sp_0,  next_q_sp_1)

                # y_aug = R_dense + γ(1-d)[min Q_aug(s',ã') − α log π(ã'|s')]
                target_aug = (
                    replay_data.rewards
                    + (1.0 - replay_data.dones) * self.gamma
                    * (next_q_aug - self.ent_coef * next_log_prob)
                )

                # y_sparse: Bootstraps the sparse signal back through the trajectory
                # This allows ∇_a Q_sparse to be non-zero during mid-flight!
                target_sparse = sparse_rewards + (1.0 - replay_data.dones) * self.gamma * next_q_sp

            # ── Critic loss ───────────────────────────────────────────
            (q_aug_0, q_aug_1), (q_sp_0, q_sp_1) = \
                self.critic.forward_decomposed(
                    replay_data.observations, replay_data.actions
                )

            loss_aug = 0.5 * (
                nn.functional.mse_loss(q_aug_0, target_aug)
                + nn.functional.mse_loss(q_aug_1, target_aug)
            )
            loss_sparse = 0.5 * (
                nn.functional.mse_loss(q_sp_0, target_sparse)
                + nn.functional.mse_loss(q_sp_1, target_sparse)
            )
            critic_loss = loss_aug + loss_sparse

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # ── Actor loss ────────────────────────────────────────────
            # Q_soft = Q_aug + Q_sparse  →  use min of each head separately
            # then sum, so clipped double-Q is applied per component.
            actions_pi, log_prob = self.actor.action_log_prob(
                replay_data.observations
            )
            (q_aug_pi_0, q_aug_pi_1), (q_sp_pi_0, q_sp_pi_1) = \
                self.critic.forward_decomposed(
                    replay_data.observations, actions_pi
                )

            min_q_aug = torch.min(q_aug_pi_0, q_aug_pi_1)  # (B, 1)
            min_q_sp  = torch.min(q_sp_pi_0,  q_sp_pi_1)   # (B, 1)
            min_q_pi  = min_q_aug + min_q_sp                # Q_soft

            actor_loss = (self.ent_coef * log_prob - min_q_pi).mean()

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # ── Entropy coefficient update (standard SAC) ─────────────
            if self.ent_coef_optimizer is not None:
                with torch.no_grad():
                    _, log_prob = self.actor.action_log_prob(
                        replay_data.observations
                    )
                ent_coef_loss = -(
                    self.log_ent_coef * (log_prob + self.target_entropy)
                ).mean()
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
                self.ent_coef = torch.exp(self.log_ent_coef.detach())

            # ── Soft update of the target critic (standard SAC) ───────
            if self._n_updates % self.target_update_interval == 0:
                for param, target_param in zip(
                    self.critic.parameters(),
                    self.critic_target.parameters(),
                ):
                    target_param.data.copy_(
                        self.tau * param.data
                        + (1.0 - self.tau) * target_param.data
                    )


# ─── 5. Exact gradient extraction for dominance analysis ─────────────────────

def compute_exact_gradients(
    model: SAC, 
    obs: np.ndarray,      # Expected shape: (Batch, Obs_Dim)
    action: np.ndarray    # Expected shape: (Batch, Act_Dim)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    model.policy.set_training_mode(False)

    # Convert to tensors
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=model.device)
    act_t = torch.as_tensor(action, dtype=torch.float32, device=model.device).requires_grad_(True)

    # Forward pass (Batch, 1)
    (q_aug_0, q_aug_1), (q_sp_0, q_sp_1) = model.critic.forward_decomposed(obs_t, act_t)

    # Clipped Double-Q (Min)
    q_aug = torch.min(q_aug_0, q_aug_1)
    q_sp  = torch.min(q_sp_0, q_sp_1)

    # Batch gradient computation
    # grad_outputs=ones allows backprop through the batch independently
    grad_aug = torch.autograd.grad(
        q_aug, act_t, grad_outputs=torch.ones_like(q_aug), retain_graph=True
    )[0].detach().cpu().numpy()

    grad_sparse = torch.autograd.grad(
        q_sp, act_t, grad_outputs=torch.ones_like(q_sp)
    )[0].detach().cpu().numpy()

    return grad_aug, grad_sparse, q_aug.detach().cpu().numpy().flatten(), q_sp.detach().cpu().numpy().flatten()

def dominance_metrics(
    grad_aug: np.ndarray,    # (Batch, Act_Dim)
    grad_sparse: np.ndarray, # (Batch, Act_Dim)
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized calculation of lambda and cos_theta."""
    
    # L2 Norms across the action dimension (axis=1)
    norm_aug = np.linalg.norm(grad_aug, axis=1, keepdims=True)
    norm_sp  = np.linalg.norm(grad_sparse, axis=1, keepdims=True)

    # Avoid division by zero
    norm_aug = np.maximum(norm_aug, 1e-9)
    
    # Lambda: Ratio of magnitudes
    lam = (norm_sp / norm_aug).flatten()

    # Cosine Similarity: (A · B) / (||A|| ||B||)
    dot_product = np.sum(grad_aug * grad_sparse, axis=1, keepdims=True)
    cos_theta = (dot_product / (norm_aug * np.maximum(norm_sp, 1e-9))).flatten()

    return lam, cos_theta