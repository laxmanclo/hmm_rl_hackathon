"""
Final Evaluation on Test Set
============================

Evaluates the trained agent on the 2000-word test set
and calculates the final score according to the formula:

Final Score = (Success Rate * 2000) - (Total Wrong Guesses * 5) - (Total Repeated Guesses * 2)
"""

import time
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from hmm_model import HangmanHMM
from hangman_env import HangmanEnv
from q_learning_agent import QLearningAgent


def evaluate_on_test_set(agent, test_words, max_wrong_guesses=6):
    """
    Evaluate agent on test set

    Args:
        agent: Trained Q-Learning agent
        test_words: List of test words
        max_wrong_guesses: Lives per game (default: 6)

    Returns:
        dict: Detailed results
    """
    print("=" * 70)
    print("EVALUATING ON TEST SET")
    print("=" * 70)
    print(f"\nTest set size: {len(test_words)} words")
    print(f"Max wrong guesses per game: {max_wrong_guesses}")

    # Create environment with test words
    env = HangmanEnv(test_words, max_wrong_guesses=max_wrong_guesses)

    # Track detailed statistics
    results = []
    total_wins = 0
    total_wrong_guesses = 0
    total_repeated_guesses = 0

    print("\nRunning evaluation...")
    start_time = time.time()

    for i, word in enumerate(test_words, 1):
        # Play one game
        result = agent.play_episode(env)

        # Count repeated guesses
        # (Environment tracks this in total_repeated_guesses)
        # For this specific game, we need to check
        repeated_in_game = env.total_repeated_guesses - total_repeated_guesses

        # Store result
        game_result = {
            'word': word,
            'won': result['won'],
            'wrong_guesses': result['wrong_guesses'],
            'repeated_guesses': repeated_in_game,
            'total_guesses': result['steps'],
            'word_length': len(word)
        }
        results.append(game_result)

        # Update totals
        if result['won']:
            total_wins += 1
        total_wrong_guesses += result['wrong_guesses']
        total_repeated_guesses = env.total_repeated_guesses

        # Progress indicator
        if i % 200 == 0:
            elapsed = time.time() - start_time
            win_rate_so_far = total_wins / i
            print(f"  Progress: {i}/{len(test_words)} games "
                  f"({100*i/len(test_words):.1f}%) - "
                  f"Win rate: {100*win_rate_so_far:.1f}% - "
                  f"Time: {elapsed:.1f}s")

    elapsed_time = time.time() - start_time

    # Calculate final metrics
    success_rate = total_wins / len(test_words)

    # OFFICIAL SCORING FORMULA from PDF
    final_score = (success_rate * 2000) - (total_wrong_guesses * 5) - (total_repeated_guesses * 2)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)
    print(f"Time taken: {elapsed_time:.1f}s ({elapsed_time/len(test_words):.3f}s per game)")

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"\nTotal games played: {len(test_words)}")
    print(f"Games won: {total_wins}")
    print(f"Games lost: {len(test_words) - total_wins}")
    print(f"\nSuccess Rate: {success_rate:.4f} = {100*success_rate:.2f}%")
    print(f"Total Wrong Guesses: {total_wrong_guesses}")
    print(f"Total Repeated Guesses: {total_repeated_guesses}")
    print(f"Average Wrong Guesses per game: {total_wrong_guesses/len(test_words):.2f}")
    print(f"Average Repeated Guesses per game: {total_repeated_guesses/len(test_words):.2f}")

    print("\n" + "=" * 70)
    print("FINAL SCORE CALCULATION")
    print("=" * 70)
    print(f"\nFormula: (Success Rate × 2000) - (Total Wrong × 5) - (Total Repeated × 2)")
    print(f"\nBreakdown:")
    print(f"  Success Rate × 2000 = {success_rate:.4f} × 2000 = {success_rate * 2000:.2f}")
    print(f"  Total Wrong × 5 = {total_wrong_guesses} × 5 = {total_wrong_guesses * 5:.2f}")
    print(f"  Total Repeated × 2 = {total_repeated_guesses} × 2 = {total_repeated_guesses * 2:.2f}")
    print(f"\n  FINAL SCORE = {success_rate * 2000:.2f} - {total_wrong_guesses * 5:.2f} - {total_repeated_guesses * 2:.2f}")
    print(f"              = {final_score:.2f}")

    print("\n" + "=" * 70)

    return {
        'results': results,
        'total_games': len(test_words),
        'total_wins': total_wins,
        'total_losses': len(test_words) - total_wins,
        'success_rate': success_rate,
        'total_wrong_guesses': total_wrong_guesses,
        'total_repeated_guesses': total_repeated_guesses,
        'avg_wrong_guesses': total_wrong_guesses / len(test_words),
        'avg_repeated_guesses': total_repeated_guesses / len(test_words),
        'final_score': final_score,
        'elapsed_time': elapsed_time
    }


def generate_detailed_plots(evaluation_results, save_path='evaluation_results.png'):
    """
    Generate comprehensive visualization of results

    Args:
        evaluation_results: Results from evaluate_on_test_set
        save_path: Where to save the plot
    """
    results = evaluation_results['results']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Test Set Evaluation Results', fontsize=16, fontweight='bold')

    # Plot 1: Win/Loss Distribution
    wins = evaluation_results['total_wins']
    losses = evaluation_results['total_losses']

    axes[0, 0].bar(['Won', 'Lost'], [wins, losses], color=['green', 'red'], alpha=0.7)
    axes[0, 0].set_ylabel('Number of Games')
    axes[0, 0].set_title(f'Win/Loss Distribution\nWin Rate: {100*evaluation_results["success_rate"]:.2f}%')
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, v in enumerate([wins, losses]):
        axes[0, 0].text(i, v + 20, str(v), ha='center', fontweight='bold')

    # Plot 2: Wrong Guesses Distribution
    wrong_guesses = [r['wrong_guesses'] for r in results]
    wrong_counts = Counter(wrong_guesses)

    x = sorted(wrong_counts.keys())
    y = [wrong_counts[i] for i in x]

    axes[0, 1].bar(x, y, color='orange', alpha=0.7)
    axes[0, 1].set_xlabel('Wrong Guesses per Game')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Wrong Guesses Distribution\nAvg: {evaluation_results["avg_wrong_guesses"]:.2f}')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Plot 3: Word Length vs Win Rate
    length_stats = {}
    for r in results:
        length = r['word_length']
        if length not in length_stats:
            length_stats[length] = {'total': 0, 'wins': 0}
        length_stats[length]['total'] += 1
        if r['won']:
            length_stats[length]['wins'] += 1

    lengths = sorted(length_stats.keys())
    win_rates = [length_stats[l]['wins'] / length_stats[l]['total'] for l in lengths]
    counts = [length_stats[l]['total'] for l in lengths]

    axes[0, 2].bar(lengths, win_rates, alpha=0.7, color='purple')
    axes[0, 2].set_xlabel('Word Length')
    axes[0, 2].set_ylabel('Win Rate')
    axes[0, 2].set_title('Win Rate by Word Length')
    axes[0, 2].set_ylim([0, 1])
    axes[0, 2].grid(True, alpha=0.3, axis='y')

    # Plot 4: Cumulative Win Rate
    cumulative_wins = []
    running_wins = 0
    for i, r in enumerate(results, 1):
        if r['won']:
            running_wins += 1
        cumulative_wins.append(running_wins / i)

    axes[1, 0].plot(cumulative_wins, linewidth=2, color='blue')
    axes[1, 0].axhline(y=evaluation_results['success_rate'], color='red',
                       linestyle='--', label=f'Final: {100*evaluation_results["success_rate"]:.2f}%')
    axes[1, 0].set_xlabel('Game Number')
    axes[1, 0].set_ylabel('Cumulative Win Rate')
    axes[1, 0].set_title('Win Rate Over Time')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 5: Total Guesses Distribution
    total_guesses = [r['total_guesses'] for r in results]

    axes[1, 1].hist(total_guesses, bins=20, color='teal', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Total Guesses per Game')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'Guesses Distribution\nAvg: {np.mean(total_guesses):.2f}')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # Plot 6: Score Breakdown
    score_components = [
        evaluation_results['success_rate'] * 2000,
        -evaluation_results['total_wrong_guesses'] * 5,
        -evaluation_results['total_repeated_guesses'] * 2
    ]
    labels = ['Success\n(+points)', 'Wrong\nGuesses\n(-points)', 'Repeated\nGuesses\n(-points)']
    colors = ['green', 'red', 'orange']

    bars = axes[1, 2].bar(labels, score_components, color=colors, alpha=0.7, edgecolor='black')
    axes[1, 2].set_ylabel('Score Contribution')
    axes[1, 2].set_title(f'Final Score Breakdown\nTotal: {evaluation_results["final_score"]:.2f}')
    axes[1, 2].axhline(y=0, color='black', linewidth=0.5)
    axes[1, 2].grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[1, 2].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.0f}',
                       ha='center', va='bottom' if height > 0 else 'top',
                       fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nEvaluation plots saved to: {save_path}")

    return fig


def save_detailed_report(evaluation_results, save_path='evaluation_report.txt'):
    """
    Save detailed text report

    Args:
        evaluation_results: Results from evaluate_on_test_set
        save_path: Where to save the report
    """
    with open(save_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("HANGMAN RL AGENT - FINAL EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write("OVERALL STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total games played: {evaluation_results['total_games']}\n")
        f.write(f"Games won: {evaluation_results['total_wins']}\n")
        f.write(f"Games lost: {evaluation_results['total_losses']}\n")
        f.write(f"Success Rate: {100*evaluation_results['success_rate']:.2f}%\n\n")

        f.write("GUESSING STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total Wrong Guesses: {evaluation_results['total_wrong_guesses']}\n")
        f.write(f"Total Repeated Guesses: {evaluation_results['total_repeated_guesses']}\n")
        f.write(f"Average Wrong Guesses per game: {evaluation_results['avg_wrong_guesses']:.2f}\n")
        f.write(f"Average Repeated Guesses per game: {evaluation_results['avg_repeated_guesses']:.2f}\n\n")

        f.write("FINAL SCORE\n")
        f.write("-" * 70 + "\n")
        f.write("Formula: (Success Rate × 2000) - (Total Wrong × 5) - (Total Repeated × 2)\n\n")
        f.write(f"Success Rate × 2000 = {evaluation_results['success_rate']:.4f} × 2000 = {evaluation_results['success_rate'] * 2000:.2f}\n")
        f.write(f"Total Wrong × 5 = {evaluation_results['total_wrong_guesses']} × 5 = {evaluation_results['total_wrong_guesses'] * 5:.2f}\n")
        f.write(f"Total Repeated × 2 = {evaluation_results['total_repeated_guesses']} × 2 = {evaluation_results['total_repeated_guesses'] * 2:.2f}\n\n")
        f.write(f"FINAL SCORE: {evaluation_results['final_score']:.2f}\n\n")

        # Sample of failed games
        failed_games = [r for r in evaluation_results['results'] if not r['won']]
        if failed_games:
            f.write("\nSAMPLE OF FAILED GAMES (first 20)\n")
            f.write("-" * 70 + "\n")
            for i, game in enumerate(failed_games[:20], 1):
                f.write(f"{i}. Word: '{game['word']}' (length {game['word_length']}) - "
                       f"Wrong: {game['wrong_guesses']}, Repeated: {game['repeated_guesses']}\n")

    print(f"Detailed report saved to: {save_path}")


def main():
    """Main evaluation pipeline"""

    # Paths
    test_path = r"C:\Users\laxma\OneDrive\Desktop\hmm_rl_hackathon\Data\Data\test.txt"
    hmm_path = r"C:\Users\laxma\OneDrive\Desktop\hmm_rl_hackathon\hmm_model.pkl"
    agent_path = r"C:\Users\laxma\OneDrive\Desktop\hmm_rl_hackathon\trained_agent.pkl"

    plot_path = r"C:\Users\laxma\OneDrive\Desktop\hmm_rl_hackathon\evaluation_results.png"
    report_path = r"C:\Users\laxma\OneDrive\Desktop\hmm_rl_hackathon\evaluation_report.txt"

    # Load test words
    print("Loading test set...")
    with open(test_path, 'r', encoding='utf-8') as f:
        test_words = [line.strip().upper() for line in f if line.strip()]
    print(f"Loaded {len(test_words)} test words\n")

    # Load HMM
    print("Loading HMM model...")
    hmm = HangmanHMM()
    hmm.load(hmm_path)

    # Load trained agent
    print("Loading trained agent...")
    agent = QLearningAgent(hmm_model=hmm)
    agent.load(agent_path)
    print(f"Agent loaded with {len(agent.q_table)} Q-table entries\n")

    # Run evaluation
    evaluation_results = evaluate_on_test_set(agent, test_words, max_wrong_guesses=6)

    # Generate visualizations
    print("\nGenerating plots...")
    generate_detailed_plots(evaluation_results, plot_path)

    # Save detailed report
    print("Generating detailed report...")
    # save_detailed_report(evaluation_results, report_path)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)
    print(f"\nFinal Score: {evaluation_results['final_score']:.2f}")
    print(f"Success Rate: {100*evaluation_results['success_rate']:.2f}%")
    print(f"\nGenerated files:")
    print(f"  - {plot_path}")
    print(f"  - {report_path}")


if __name__ == "__main__":
    main()
