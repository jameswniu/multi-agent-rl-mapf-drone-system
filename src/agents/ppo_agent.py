"""
PPO Agent with Integrity Validation
-----------------------------------
This file defines a Proximal Policy Optimization (PPO) agent in PyTorch.

Why PPO?
-> PPO is one of the most widely used reinforcement learning algorithms.
-> It improves stability using a "clipping trick" that prevents huge policy updates.
-> It separates policy (actions) from value (baseline), which reduces variance.

Why integrity validation?
-> After every policy decision, we check the probabilities, value estimates,
   and chosen actions to make sure they are valid.
-> This prevents silent bugs (like NaNs, negative probabilities, or invalid actions).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

from integrity_validators import PolicyIntegrityValidator  # schema-based validator


# ---------------- Policy Network ----------------

class PPOPolicy(nn.Module):
    """
    The policy and value networks share a common backbone.

    - Shared layers -> extract features from the state.
    - Policy head -> outputs action probabilities.
    - Value head -> predicts the baseline value of the state.

    This "actor-critic" design is standard in modern RL.
    """

    def __init__(self, obs_dim, act_dim):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
        )

        # Policy head: probability distribution over actions
        self.policy_head = nn.Sequential(
            nn.Linear(64, act_dim),
            nn.Softmax(dim=-1),  # ensures outputs are valid probabilities
        )

        # Value head: scalar baseline for this state
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        features = self.shared(x)
        action_probs = self.policy_head(features)
        value = self.value_head(features)
        return action_probs, value


# ---------------- PPO Agent ----------------

class PPOAgent:
    """
    Encapsulates PPO training and inference.

    Main methods:
    - select_action(state) -> sample an action for exploration
    - train(num_episodes) -> run episodes and update policy
    - predict(state) -> pick greedy action (for inference/production)
    - save(path)/load(path) -> persist model weights
    """

    def __init__(self, env, lr=3e-4, gamma=0.99, eps_clip=0.2, epochs=3):
        self.env = env
        self.gamma = gamma       # discount factor
        self.eps_clip = eps_clip # PPO clipping parameter
        self.epochs = epochs     # policy update iterations

        obs_dim = env.observation_space.shape[0]  # 5 features
        act_dim = env.action_space.n              # 5 actions

        # Initialize policy network and optimizer
        self.policy = PPOPolicy(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Attach validator for drift/hallucination checks
        self.validator = PolicyIntegrityValidator(env.action_space)

    def select_action(self, state):
        """
        Pick an action from the current policy.
        -> During training, we sample from the distribution (exploration).
        """
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probs, value = self.policy(state)
        m = Categorical(probs)
        action = m.sample()

        # Integrity check
        errors = self.validator.validate(probs.squeeze(), value.squeeze(), action.item())
        if errors:
            for e in errors:
                print(f"[Integrity Warning] {e['type']} on {e['field']}: {e['msg']}")

        return action.item(), m.log_prob(action)

    def train(self, num_episodes=100):
        """
        Train the PPO agent for a number of episodes.

        Training loop:
        - Run one episode, collecting (state, action, reward).
        - Compute discounted returns.
        - Update policy using clipped surrogate objective.
        """
        for ep in range(num_episodes):
            state, _ = self.env.reset()
            log_probs, rewards, states, actions = [], [], [], []
            terminated, truncated = False, False

            # Rollout one episode
            while not (terminated or truncated):
                action, log_prob = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)

                state = next_state

            # Compute discounted returns
            discounted = []
            R = 0
            for r in reversed(rewards):
                R = r + self.gamma * R
                discounted.insert(0, R)
            discounted = torch.tensor(discounted, dtype=torch.float32)

            # Normalize returns -> improves training stability
            discounted = (discounted - discounted.mean()) / (discounted.std() + 1e-8)

            # Policy update for several epochs
            for _ in range(self.epochs):
                states_t = torch.tensor(np.array(states), dtype=torch.float32)
                actions_t = torch.tensor(actions, dtype=torch.long)
                old_log_probs = torch.stack(log_probs)

                # Forward pass
                probs, values = self.policy(states_t)
                m = Categorical(probs)
                new_log_probs = m.log_prob(actions_t)
                entropy = m.entropy().mean()  # encourages exploration

                # Advantage = return - baseline
                advantages = discounted - values.squeeze()
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # PPO surrogate loss
                ratio = (new_log_probs - old_log_probs.detach()).exp()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                # Value loss (MSE)
                value_loss = (discounted - values.squeeze()) ** 2

                # Total loss = policy + value - entropy
                loss = -torch.min(surr1, surr2).mean() \
                       + 0.5 * value_loss.mean() \
                       - 0.01 * entropy

                # Gradient update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f"Episode {ep+1}, total reward={sum(rewards):.2f}")

    def save(self, path):
        """Save model weights to disk."""
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        """Load model weights from disk."""
        self.policy.load_state_dict(torch.load(path))

    def predict(self, state):
        """
        Greedy action selection (for inference).
        -> Use after training when running in production.
        """
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probs, value = self.policy(state)
        action = torch.argmax(probs, dim=-1).item()

        # Integrity check
        errors = self.validator.validate(probs.squeeze(), value.squeeze(), action)
        if errors:
            for e in errors:
                print(f"[Integrity Warning] {e['type']} on {e['field']}: {e['msg']}")

        return action
