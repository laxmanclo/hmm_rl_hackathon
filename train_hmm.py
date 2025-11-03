"""
HMM Training Script
==================

Train the Hidden Markov Model on the cleaned corpus and test it
"""

import time
from hmm_model import HangmanHMM


def test_hmm(hmm, test_cases):
    """
    Test HMM with sample partial words

    Args:
        hmm: Trained HMM model
        test_cases: List of (masked_word, guessed_letters) tuples
    """
    print("\n" + "=" * 70)
    print("TESTING HMM PREDICTIONS")
    print("=" * 70)

    for masked_word, guessed_letters in test_cases:
        print(f"\nTest case: '{masked_word}'")
        print(f"Already guessed: {sorted(guessed_letters)}")

        probs = hmm.get_letter_probabilities(masked_word, guessed_letters)

        # Show top 10 predictions
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:10]

        print("Top 10 letter predictions:")
        for i, (letter, prob) in enumerate(sorted_probs, 1):
            bar = '#' * int(prob * 50)
            print(f"  {i:2d}. {letter}: {prob:.4f} {bar}")


def main():
    """Train and test the HMM"""

    # Paths
    corpus_path = r"C:\Users\laxma\OneDrive\Desktop\hmm_rl_hackathon\corpus_cleaned.txt"
    model_save_path = r"C:\Users\laxma\OneDrive\Desktop\hmm_rl_hackathon\hmm_model.pkl"

    # Create and train HMM
    print("=" * 70)
    print("TRAINING HIDDEN MARKOV MODEL")
    print("=" * 70)

    hmm = HangmanHMM()

    start_time = time.time()
    hmm.train(corpus_path)
    training_time = time.time() - start_time

    print(f"\nTraining completed in {training_time:.2f} seconds")

    # Save model
    hmm.save(model_save_path)

    # Test cases (simulating Hangman game states)
    test_cases = [
        # (masked_word, guessed_letters)
        ("_____", set()),  # Fresh start, no guesses
        ("_____", {'E', 'T', 'A'}),  # Common letters tried
        ("_PP__", {'P'}),  # Pattern: _PP__
        ("_PP_E", {'P', 'E'}),  # Pattern: _PP_E
        ("____ING", {'I', 'N', 'G'}),  # Common ending
        ("TH_", {'T', 'H'}),  # Short word
        ("__LL_", {'L'}),  # Double L pattern
        ("___E_", {'E'}),  # E in middle
        ("_____ION", {'I', 'O', 'N'}),  # -TION ending likely
        ("PR_____", {'P', 'R'}),  # PR- start
    ]

    test_hmm(hmm, test_cases)

    print("\n" + "=" * 70)
    print("HMM READY FOR USE")
    print("=" * 70)
    print(f"\nModel saved to: {model_save_path}")
    print("You can now use this model in your RL agent!")


if __name__ == "__main__":
    main()
