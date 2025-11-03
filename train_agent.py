"""
Training Script for Hangman RL Agent
====================================

This script:
1. Loads the trained HMM
2. Creates Hangman environment with corpus words
3. Trains Q-Learning agent
4. Tracks and visualizes progress
5. Saves trained agent
"""

import time
import matplotlib.pyplot as plt
from hmm_model import HangmanHMM
from hangman_env import HangmanEnv
from q_learning_agent import QLearningAgent


def plot_training_progress(agent, env_stats, save_path='training_progress.png'):
    """
    Plot training metrics

    Args:
        agent: Trained agent
        env_stats: Environment statistics
        save_path: Where to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Q-Learning Training Progress', fontsize=16)

    # Plot 1: Episode Rewards
    if agent.episode_rewards:
        axes[0, 0].plot(agent.episode_rewards, alpha=0.3, color='blue')
        # Moving average
        window = 100
        if len(agent.episode_rewards) >= window:
            moving_avg = []
            for i in range(len(agent.episode_rewards) - window + 1):
                moving_avg.append(sum(agent.episode_rewards[i:i+window]) / window)
            axes[0, 0].plot(range(window-1, len(agent.episode_rewards)), moving_avg,
                           color='red', linewidth=2, label=f'{window}-episode moving avg')
            axes[0, 0].legend()

        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].set_title('Reward per Episode')
        axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Episode Lengths
    if agent.episode_lengths:
        axes[0, 1].plot(agent.episode_lengths, alpha=0.3, color='green')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps per Episode')
        axes[0, 1].set_title('Episode Length (Number of Guesses)')
        axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Win Rate Over Time
    if agent.episode_rewards:
        window = 100
        win_rates = []
        wins_in_window = 0

        for i, reward in enumerate(agent.episode_rewards):
            # Simple heuristic: positive reward = win
            if reward > 50:
                wins_in_window += 1

            if i >= window:
                # Remove old episode from window
                if agent.episode_rewards[i - window] > 50:
                    wins_in_window -= 1
                win_rates.append(wins_in_window / window)

        if win_rates:
            axes[1, 0].plot(range(window, len(agent.episode_rewards)), win_rates,
                           color='purple', linewidth=2)
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Win Rate')
            axes[1, 0].set_title(f'Win Rate ({window}-episode window)')
            axes[1, 0].set_ylim([0, 1])
            axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Epsilon Decay
    epsilon_values = []
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_end = 0.01

    for _ in range(len(agent.episode_rewards)):
        epsilon_values.append(epsilon)
        epsilon = max(epsilon * epsilon_decay, epsilon_end)

    axes[1, 1].plot(epsilon_values, color='orange', linewidth=2)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Epsilon (Exploration Rate)')
    axes[1, 1].set_title('Exploration Rate Decay')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nTraining plot saved to: {save_path}")

    return fig


def train_agent(
    hmm_model,
    words,
    num_episodes=10000,
    eval_every=1000,
    save_every=2000,
    agent_save_path='trained_agent.pkl',
    plot_save_path='training_progress.png'
):
    """
    Main training loop

    Args:
        hmm_model: Trained HMM
        words: List of words for training
        num_episodes: Number of training games
        eval_every: Evaluate every N episodes
        save_every: Save agent every N episodes
        agent_save_path: Where to save trained agent
        plot_save_path: Where to save training plot
    """
    print("=" * 70)
    print("TRAINING Q-LEARNING AGENT")
    print("=" * 70)

    # Create environment
    env = HangmanEnv(words)

    # Create agent
    agent = QLearningAgent(
        hmm_model=hmm_model,
        alpha=0.1,           # Learning rate
        gamma=0.95,          # Discount factor
        epsilon_start=1.0,   # Start with full exploration
        epsilon_end=0.01,    # End with 1% exploration
        epsilon_decay=0.995  # Decay per episode
    )

    print(f"\nTraining configuration:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Learning rate (alpha): {agent.alpha}")
    print(f"  Discount factor (gamma): {agent.gamma}")
    print(f"  Epsilon decay: {agent.epsilon_decay}")
    print(f"  Vocabulary size: {len(words)} words")

    # Training loop
    print(f"\nStarting training...")
    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        # Train one episode
        episode_stats = agent.train_episode(env)

        # Track statistics
        agent.episode_rewards.append(episode_stats['total_reward'])
        agent.episode_lengths.append(episode_stats['steps'])

        # Periodic evaluation
        if episode % eval_every == 0:
            elapsed_time = time.time() - start_time

            # Calculate recent performance
            recent_window = 100
            if len(agent.episode_rewards) >= recent_window:
                recent_rewards = agent.episode_rewards[-recent_window:]
                avg_reward = sum(recent_rewards) / len(recent_rewards)

                recent_wins = sum(1 for r in recent_rewards if r > 50)
                win_rate = recent_wins / recent_window
            else:
                avg_reward = sum(agent.episode_rewards) / len(agent.episode_rewards)
                win_rate = 0.0

            print(f"\nEpisode {episode}/{num_episodes}")
            print(f"  Time: {elapsed_time:.1f}s")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Avg reward (last {recent_window}): {avg_reward:.2f}")
            print(f"  Win rate (last {recent_window}): {win_rate:.2%}")
            print(f"  Q-table size: {len(agent.q_table)} entries")

        # Periodic saving
        if episode % save_every == 0:
            agent.save(agent_save_path)
            plot_training_progress(agent, env.get_statistics(), plot_save_path)

    # Final save and evaluation
    training_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Training time: {training_time:.1f}s ({training_time/num_episodes:.3f}s per episode)")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Q-table size: {len(agent.q_table)} entries")

    # Save final agent
    agent.save(agent_save_path)

    # Generate final plot
    plot_training_progress(agent, env.get_statistics(), plot_save_path)

    # Final evaluation on 100 games
    print("\n" + "=" * 70)
    print("FINAL EVALUATION (100 games)")
    print("=" * 70)

    eval_env = HangmanEnv(words)
    eval_results = []

    for _ in range(100):
        result = agent.play_episode(eval_env)
        eval_results.append(result)

    wins = sum(1 for r in eval_results if r['won'])
    total_wrong = sum(r['wrong_guesses'] for r in eval_results)
    avg_wrong = total_wrong / 100

    print(f"\nFinal Performance:")
    print(f"  Win rate: {wins}/100 = {wins}%")
    print(f"  Avg wrong guesses: {avg_wrong:.2f}")
    print(f"  Total wrong guesses: {total_wrong}")

    return agent, env


def main():
    """Main training pipeline"""

    # Paths
    corpus_path = r"C:\Users\laxma\OneDrive\Desktop\hmm_rl_hackathon\corpus_cleaned.txt"
    hmm_path = r"C:\Users\laxma\OneDrive\Desktop\hmm_rl_hackathon\hmm_model.pkl"
    agent_save_path = r"C:\Users\laxma\OneDrive\Desktop\hmm_rl_hackathon\trained_agent.pkl"
    plot_save_path = r"C:\Users\laxma\OneDrive\Desktop\hmm_rl_hackathon\training_progress.png"

    # Load HMM
    print("Loading HMM model...")
    hmm = HangmanHMM()
    hmm.load(hmm_path)
    print("[OK] HMM loaded")

    # Load corpus
    print("\nLoading training corpus...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()]
    print(f"[OK] Loaded {len(words)} words")

    # Train agent
    agent, env = train_agent(
        hmm_model=hmm,
        words=words,
        num_episodes=30000,  # Adjust based on time available
        eval_every=500,
        save_every=2000,
        agent_save_path=agent_save_path,
        plot_save_path=plot_save_path
    )

    print("\n" + "=" * 70)
    print("READY FOR TESTING!")
    print("=" * 70)
    print(f"\nTrained agent saved to: {agent_save_path}")
    print(f"Training plot saved to: {plot_save_path}")
    print("\nNext step: Evaluate on test set (test.txt)")


if __name__ == "__main__":
    main()
