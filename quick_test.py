"""
Quick Test of Improvements
==========================

Test the improved HMM + RL on a small sample (100 games)
to see if performance improved before full retraining
"""

from hmm_model import HangmanHMM
from hangman_env import HangmanEnv
from q_learning_agent import QLearningAgent

def quick_test(test_words, max_games=100):
    """Run quick test on sample"""
    print("=" * 70)
    print(f"QUICK TEST - {max_games} Games")
    print("=" * 70)

    # Load improved HMM
    print("\nLoading improved HMM...")
    hmm = HangmanHMM()
    hmm.load(r"C:\Users\laxma\OneDrive\Desktop\hmm_rl_hackathon\hmm_model.pkl")

    # Create new agent (untrained, just using improved state representation)
    print("Creating agent with improved state representation...")
    agent = QLearningAgent(hmm_model=hmm)

    # Test on sample
    test_sample = test_words[:max_games]
    env = HangmanEnv(test_sample, max_wrong_guesses=6)

    wins = 0
    total_wrong = 0

    print(f"\nRunning {max_games} games...")
    for i, word in enumerate(test_sample, 1):
        result = agent.play_episode(env)
        if result['won']:
            wins += 1
        total_wrong += result['wrong_guesses']

        if i % 20 == 0:
            print(f"  Progress: {i}/{max_games} - Win rate: {100*wins/i:.1f}%")

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Win rate: {100*wins/max_games:.1f}% ({wins}/{max_games})")
    print(f"Avg wrong guesses: {total_wrong/max_games:.2f}")
    print("\nNote: This is with UNTRAINED agent using improved features")
    print("After training, we expect much better performance!")
    print("=" * 70)

if __name__ == "__main__":
    # Load test words
    test_path = r"C:\Users\laxma\OneDrive\Desktop\hmm_rl_hackathon\Data\Data\test.txt"
    with open(test_path, 'r', encoding='utf-8') as f:
        test_words = [line.strip().upper() for line in f if line.strip()]

    quick_test(test_words, max_games=100)
