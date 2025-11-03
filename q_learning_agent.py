"""
Q-Learning Agent for Hangman
============================

This implements the RL "brain" that learns optimal letter-guessing strategy.

Q-LEARNING MATHEMATICS:
----------------------

Q-Value: Q(s, a) = "How good is it to take action a in state s?"

Bellman Equation:
    Q(s, a) ← Q(s, a) + α × [r + γ × max_a' Q(s', a') - Q(s, a)]

Breaking it down:
- α (alpha): Learning rate (0 < α ≤ 1)
    - How much to update Q-value based on new experience
    - α = 0.1 means "update 10% toward new information"

- r: Immediate reward from action

- γ (gamma): Discount factor (0 < γ ≤ 1)
    - How much future rewards matter
    - γ = 0.9 means "future reward is worth 90% of immediate reward"

- max_a' Q(s', a'): Best Q-value possible from next state s'

- The bracketed part [r + γ × max - Q(s,a)] is the "TD error"
    - How much our prediction was wrong
    - Positive = action was better than expected
    - Negative = action was worse than expected

EXPLORATION vs EXPLOITATION:
----------------------------
ε-greedy strategy:
- With probability ε: Explore (random action)
- With probability (1-ε): Exploit (best known action)
- ε decays over time: start exploring, end up exploiting

Example:
- Episode 1: ε = 0.9 → 90% random guesses (learn new things)
- Episode 5000: ε = 0.01 → 1% random guesses (use what we learned)
"""

import random
import pickle
import numpy as np
from collections import defaultdict
from typing import Set, Tuple


class QLearningAgent:
    """
    Q-Learning agent that learns to play Hangman

    Uses HMM letter probabilities as part of state representation
    """

    def __init__ (
        self,
        hmm_model=None,
        alpha=0.1,          # Learning rate
        gamma=0.95,         # Discount factor
        epsilon_start=1.0,  # Initial exploration rate
        epsilon_end=0.01,   # Final exploration rate
        epsilon_decay=0.995 # Decay rate per episode
    ):
        """
        Initialize Q-Learning agent

        Args:
            hmm_model: Trained HMM for letter probability predictions
            alpha: Learning rate (how much to update Q-values)
            gamma: Discount factor (how much future rewards matter)
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon per episode
        """
        self.hmm_model = hmm_model

        # Q-learning parameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Q-table: stores Q(s, a) values
        # Key: (state_representation, action)
        # Value: Q-value (float)
        self.q_table = defaultdict(float)

        # Statistics
        self.episode_rewards = []
        self.episode_lengths = []


    def get_state_representation(self, state, hmm_probs=None):
        """
        Convert game state to a hashable representation

        STATE DESIGN CHALLENGE:
        ----------------------
        We can't use the full state (too many combinations!)

        Example: masked_word="_PP_E", guessed={P,E}, lives=5
        - Storing Q-values for EVERY possible combination = impossible

        SOLUTION: Simplify state representation with informative features
        - Word length and revealed ratio
        - Lives category
        - Pattern features (has doubles, has endings, etc.)
        - HMM's top predictions

        This reduces state space while keeping important information!

        Args:
            state: Game state dict
            hmm_probs: HMM probability distribution (optional)

        Returns:
            tuple: Hashable state representation
        """
        masked_word = state['masked_word']
        lives_left = state['lives_left']
        num_guessed = len(state['guessed_letters'])

        # Feature 1: Word length (bucketed for generalization)
        word_length = len(masked_word)
        if word_length <= 4:
            length_bucket = 'short'
        elif word_length <= 7:
            length_bucket = 'medium'
        elif word_length <= 10:
            length_bucket = 'long'
        else:
            length_bucket = 'verylong'

        # Feature 2: Revealed ratio (how much is revealed)
        revealed_count = sum(1 for c in masked_word if c != '_')
        revealed_ratio = revealed_count / word_length if word_length > 0 else 0
        if revealed_ratio < 0.2:
            revealed_stage = 'early'
        elif revealed_ratio < 0.5:
            revealed_stage = 'mid'
        elif revealed_ratio < 0.8:
            revealed_stage = 'late'
        else:
            revealed_stage = 'endgame'

        # Feature 3: Lives category (more granular)
        if lives_left >= 5:
            life_category = 'safe'
        elif lives_left >= 3:
            life_category = 'medium'
        elif lives_left >= 1:
            life_category = 'danger'
        else:
            life_category = 'dead'

        # Feature 4: Pattern features
        has_doubles = any(masked_word[i] == masked_word[i+1] != '_'
                         for i in range(len(masked_word) - 1))

        # Check for common endings (if revealed at end)
        has_ending_pattern = False
        if len(masked_word) >= 3:
            ending = masked_word[-3:]
            if '_' not in ending and ending in ['ING', 'ION', 'TED', 'LLY', 'EST']:
                has_ending_pattern = True

        # Feature 5: HMM's top 2 recommendations (stronger signal)
        top_hmm_letter = None
        second_hmm_letter = None
        if hmm_probs and len(hmm_probs) > 0:
            sorted_probs = sorted(hmm_probs.items(), key=lambda x: x[1], reverse=True)
            top_hmm_letter = sorted_probs[0][0] if len(sorted_probs) > 0 else None
            second_hmm_letter = sorted_probs[1][0] if len(sorted_probs) > 1 else None

        # Create SIMPLER state tuple (proven to work)
        state_repr = (
            length_bucket,
            revealed_stage,
            life_category,
            has_doubles,
            has_ending_pattern,
            top_hmm_letter,
            second_hmm_letter
        )

        return state_repr


    def get_q_value(self, state_repr, action):
        """
        Get Q-value for state-action pair

        Args:
            state_repr: State representation (tuple)
            action: Letter to guess

        Returns:
            float: Q-value (defaults to 0 if never seen)
        """
        return self.q_table[(state_repr, action)]


    def get_best_action(self, state, available_actions, hmm_probs=None):
        """
        Get best action according to Q-table

        Args:
            state: Game state
            available_actions: Set of unguessed letters
            hmm_probs: HMM probability distribution

        Returns:
            str: Best letter to guess
        """
        if not available_actions:
            return None

        state_repr = self.get_state_representation(state, hmm_probs)

        # Find action with highest Q-value
        best_action = None
        best_q_value = float('-inf')

        for action in available_actions:
            q_value = self.get_q_value(state_repr, action)

            # Bonus: Add HMM probability to help tie-breaking
            if hmm_probs and action in hmm_probs:
                q_value += 0.1 * hmm_probs[action]  # Small HMM bias

            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action

        return best_action


    def choose_action(self, state, available_actions, hmm_probs=None, training=True):
        """
        Choose action using ε-greedy strategy

        EPSILON-GREEDY EXPLAINED:
        ------------------------
        Imagine you're at a restaurant:
        - Exploitation: Always order your favorite dish (safe, but miss new dishes)
        - Exploration: Try new dishes (risky, but might find something better)
        - ε-greedy: With probability ε, try something new; else, stick to favorite

        At start (ε=1.0): Always explore (learn the game)
        At end (ε=0.01): Mostly exploit (use what you learned)

        Args:
            state: Game state
            available_actions: Set of unguessed letters
            hmm_probs: HMM probability distribution
            training: Whether in training mode (affects exploration)

        Returns:
            str: Letter to guess
        """
        if not available_actions:
            return None

        # Exploration: Random action
        if training and random.random() < self.epsilon:
            # Use HMM-weighted random selection (better than uniform random)
            if hmm_probs and len(hmm_probs) > 0:
                # Sample proportional to HMM probabilities
                letters = list(available_actions)
                weights = [hmm_probs.get(l, 1e-10) for l in letters]
                total_weight = sum(weights)
                weights = [w / total_weight for w in weights]
                return np.random.choice(letters, p=weights)
            else:
                return random.choice(list(available_actions))

        # Exploitation: Best action according to Q-table + HMM
        return self.get_best_action(state, available_actions, hmm_probs)


    def update_q_value(self, state, action, reward, next_state, done, hmm_probs=None, next_hmm_probs=None):
        """
        Update Q-value using Bellman equation

        THE HEART OF Q-LEARNING:
        -----------------------

        Q(s, a) ← Q(s, a) + α × [r + γ × max_a' Q(s', a') - Q(s, a)]

        Step by step:
        1. Get current Q(s, a)
        2. Calculate target: r + γ × max Q(s', a')
        3. Calculate TD error: target - current
        4. Update: current + α × TD_error

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: State after action
            done: Whether episode ended
            hmm_probs: HMM probs for current state
            next_hmm_probs: HMM probs for next state
        """
        # Get state representations
        state_repr = self.get_state_representation(state, hmm_probs)
        next_state_repr = self.get_state_representation(next_state, next_hmm_probs)

        # Current Q-value
        current_q = self.get_q_value(state_repr, action)

        # Calculate target Q-value
        if done:
            # If episode ended, there's no future reward
            target_q = reward
        else:
            # Find best Q-value from next state
            available_next_actions = next_state.get('available_actions', set())

            if available_next_actions:
                max_next_q = max(
                    self.get_q_value(next_state_repr, a)
                    for a in available_next_actions
                )
            else:
                max_next_q = 0

            # Bellman equation: target = reward + γ × max future Q
            target_q = reward + self.gamma * max_next_q

        # TD error (how much we were wrong)
        td_error = target_q - current_q

        # Update Q-value: Q ← Q + α × TD_error
        new_q = current_q + self.alpha * td_error

        self.q_table[(state_repr, action)] = new_q


    def decay_epsilon(self):
        """
        Decay exploration rate

        As agent learns, reduce exploration:
        ε ← max(ε × decay, ε_min)
        """
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)


    def train_episode(self, env, render=False):
        """
        Train for one episode (one game)

        Returns:
            dict: Episode statistics
        """
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            # Get HMM predictions
            hmm_probs = None
            if self.hmm_model:
                available_actions = env.get_available_actions()
                if available_actions:
                    hmm_probs = self.hmm_model.get_letter_probabilities(
                        state['masked_word'],
                        state['guessed_letters']
                    )

            # Choose action
            available_actions = env.get_available_actions()
            action = self.choose_action(state, available_actions, hmm_probs, training=True)

            if action is None:
                break

            # Take action
            next_state, reward, done, info = env.step(action)

            # Get HMM predictions for next state
            next_hmm_probs = None
            if self.hmm_model and not done:
                next_available = env.get_available_actions()
                if next_available:
                    next_hmm_probs = self.hmm_model.get_letter_probabilities(
                        next_state['masked_word'],
                        next_state['guessed_letters']
                    )

            # Update Q-value
            next_state['available_actions'] = env.get_available_actions()
            self.update_q_value(state, action, reward, next_state, done, hmm_probs, next_hmm_probs)

            # Update state
            state = next_state
            total_reward += reward
            steps += 1

            if render:
                env.render()

        # Decay exploration
        self.decay_epsilon()

        return {
            'total_reward': total_reward,
            'steps': steps,
            'won': env.wrong_guesses < env.max_wrong_guesses,
            'wrong_guesses': env.wrong_guesses
        }


    def play_episode(self, env, render=False):
        """
        Play one episode without learning (evaluation mode)

        Returns:
            dict: Episode statistics
        """
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            # Get HMM predictions
            hmm_probs = None
            if self.hmm_model:
                available_actions = env.get_available_actions()
                if available_actions:
                    hmm_probs = self.hmm_model.get_letter_probabilities(
                        state['masked_word'],
                        state['guessed_letters']
                    )

            # Choose best action (no exploration)
            available_actions = env.get_available_actions()
            action = self.choose_action(state, available_actions, hmm_probs, training=False)

            if action is None:
                break

            # Take action
            state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1

            if render:
                env.render()

        return {
            'total_reward': total_reward,
            'steps': steps,
            'won': env.wrong_guesses < env.max_wrong_guesses,
            'wrong_guesses': env.wrong_guesses,
            'word': env.target_word
        }


    def save(self, filepath):
        """Save agent to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': dict(self.q_table),
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay': self.epsilon_decay,
                'episode_rewards': self.episode_rewards,
                'episode_lengths': self.episode_lengths
            }, f)
        print(f"Agent saved to {filepath}")


    def load(self, filepath):
        """Load agent from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.q_table = defaultdict(float, data['q_table'])
        self.alpha = data['alpha']
        self.gamma = data['gamma']
        self.epsilon = data['epsilon']
        self.epsilon_end = data['epsilon_end']
        self.epsilon_decay = data['epsilon_decay']
        self.episode_rewards = data.get('episode_rewards', [])
        self.episode_lengths = data.get('episode_lengths', [])
        print(f"Agent loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    print("Q-Learning Agent module loaded successfully!")
    print("\nNext step: Train the agent on the corpus")
