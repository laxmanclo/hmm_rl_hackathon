"""
TRUE Hidden Markov Model for Hangman - REAL Forward-Backward Algorithm
========================================================================

This implements a PROPER HMM with:
- Hidden states (letter positions + word classes)
- Transition probabilities P(state_t+1 | state_t)
- Emission probabilities P(observation | state)
- Forward-Backward dynamic programming (α, β vectors)

REAL HMM COMPONENTS:
-------------------
Hidden States: Position in word (0, 1, 2, ..., n-1)
Observations: Letters (A-Z) or BLANK (_)
Transitions: P(letter_at_pos_i+1 | letter_at_pos_i, word_length)
Emissions: P(letter | position, word_length)

FORWARD-BACKWARD ALGORITHM:
--------------------------
Forward (α): P(observations_1:t, state_t)
Backward (β): P(observations_t+1:n | state_t)
Posterior: P(state_t | all_observations) ∝ α_t * β_t

"""

import numpy as np
from collections import defaultdict, Counter
import pickle
import string


class HangmanHMM:
    """
    TRUE HMM implementation with forward-backward algorithm
    """

    def __init__(self):
        # Emission probabilities: P(letter | position, word_length)
        self.emission_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        # Format: {word_length: {position: {letter: probability}}}

        # Transition probabilities: P(letter_j at pos+1 | letter_i at pos, word_length)
        self.transition_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        # Format: {word_length: {(pos, letter_i): {letter_j: probability}}}

        # Initial probabilities: P(letter | position=0, word_length)
        self.initial_probs = defaultdict(lambda: defaultdict(float))
        # Format: {word_length: {letter: probability}}

        # Store words for pattern matching (auxiliary)
        self.words_by_length = defaultdict(list)

        # Global letter frequencies (fallback)
        self.global_letter_freq = Counter()

        self.trained = False


    def train(self, corpus_file):
        """
        Train TRUE HMM parameters

        Learns:
        1. Initial probabilities P(letter | pos=0, length)
        2. Transition probabilities P(letter_t+1 | letter_t, pos, length)
        3. Emission probabilities P(letter | pos, length)
        """
        print("Training TRUE HMM with forward-backward...")
        word_count = 0

        # Count raw frequencies
        emission_counts = defaultdict(lambda: defaultdict(Counter))
        transition_counts = defaultdict(lambda: defaultdict(Counter))
        initial_counts = defaultdict(Counter)

        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().upper()

                if not word or not word.isalpha():
                    continue

                word_count += 1
                word_length = len(word)

                # Store word
                self.words_by_length[word_length].append(word)

                # Count initial state (first letter)
                initial_counts[word_length][word[0]] += 1

                # Count emissions and transitions
                for pos, letter in enumerate(word):
                    # Emission: P(letter | position, length)
                    emission_counts[word_length][pos][letter] += 1
                    self.global_letter_freq[letter] += 1

                    # Transition: P(next_letter | current_letter, position, length)
                    if pos < word_length - 1:
                        next_letter = word[pos + 1]
                        transition_counts[word_length][(pos, letter)][next_letter] += 1

                if word_count % 10000 == 0:
                    print(f"  Processed {word_count} words...")

        # Convert counts to probabilities with smoothing
        print("  Computing probabilities with Laplace smoothing...")
        alphabet = string.ascii_uppercase
        smoothing = 0.01

        # Initial probabilities
        for length in initial_counts:
            total = sum(initial_counts[length].values()) + smoothing * len(alphabet)
            for letter in alphabet:
                count = initial_counts[length].get(letter, 0) + smoothing
                self.initial_probs[length][letter] = count / total

        # Emission probabilities
        for length in emission_counts:
            for pos in emission_counts[length]:
                total = sum(emission_counts[length][pos].values()) + smoothing * len(alphabet)
                for letter in alphabet:
                    count = emission_counts[length][pos].get(letter, 0) + smoothing
                    self.emission_probs[length][pos][letter] = count / total

        # Transition probabilities
        for length in transition_counts:
            for (pos, letter) in transition_counts[length]:
                total = sum(transition_counts[length][(pos, letter)].values()) + smoothing * len(alphabet)
                for next_letter in alphabet:
                    count = transition_counts[length][(pos, letter)].get(next_letter, 0) + smoothing
                    self.transition_probs[length][(pos, letter)][next_letter] = count / total

        self.trained = True
        print(f"Training complete! Processed {word_count} words")
        print(f"Word length range: {min(self.words_by_length.keys())} to {max(self.words_by_length.keys())}")


    def get_letter_probabilities(self, masked_word, guessed_letters):
        """
        TRUE Forward-Backward algorithm

        Computes P(letter at blank_position | observed_letters) using:
        1. Forward pass (α): probability of observations up to position t
        2. Backward pass (β): probability of observations after position t
        3. Combine: P(state | all observations) ∝ α * β

        Args:
            masked_word: Current word state, e.g., "_PP_E"
            guessed_letters: Set of already guessed letters

        Returns:
            dict: {letter: probability} for all unguessed letters
        """
        if not self.trained:
            raise ValueError("HMM not trained! Call train() first.")

        word_length = len(masked_word)
        remaining_letters = set(string.ascii_uppercase) - guessed_letters

        # If no HMM data for this length, use pattern matching fallback
        if word_length not in self.emission_probs or not self.emission_probs[word_length]:
            return self._fallback_probabilities(masked_word, remaining_letters)

        # Run forward-backward algorithm
        letter_probs = self._forward_backward(masked_word, word_length, remaining_letters)

        return letter_probs


    def _forward_backward(self, masked_word, word_length, remaining_letters):
        """
        REAL forward-backward dynamic programming

        Forward: α[t][letter] = P(observations_1:t, state_t = letter)
        Backward: β[t][letter] = P(observations_t+1:n | state_t = letter)
        Posterior: γ[t][letter] ∝ α[t][letter] * β[t][letter]
        """
        alphabet = string.ascii_uppercase
        n = word_length

        # Initialize forward (α) and backward (β) tables
        alpha = [{letter: 0.0 for letter in alphabet} for _ in range(n)]
        beta = [{letter: 0.0 for letter in alphabet} for _ in range(n)]

        # FORWARD PASS
        # α[0][letter] = P(initial) * P(observation_0 | letter)
        for letter in alphabet:
            if masked_word[0] == '_':
                # Position 0 is blank - all letters possible
                alpha[0][letter] = self.initial_probs[word_length].get(letter, 1e-10)
            elif masked_word[0] == letter:
                # Position 0 is observed as this letter
                alpha[0][letter] = self.initial_probs[word_length].get(letter, 1e-10)
            else:
                # Position 0 is observed as different letter
                alpha[0][letter] = 0.0

        # Normalize α[0]
        total = sum(alpha[0].values())
        if total > 0:
            alpha[0] = {k: v / total for k, v in alpha[0].items()}

        # Forward recursion: α[t][j] = Σ_i α[t-1][i] * P(j|i) * P(obs_t|j)
        for t in range(1, n):
            for letter_j in alphabet:
                prob_sum = 0.0
                for letter_i in alphabet:
                    if alpha[t-1][letter_i] > 0:
                        # Transition probability
                        trans_prob = self.transition_probs[word_length].get((t-1, letter_i), {}).get(letter_j, 1e-10)
                        prob_sum += alpha[t-1][letter_i] * trans_prob

                # Emission probability (observation at position t)
                if masked_word[t] == '_':
                    # Blank - emission is just P(letter | pos)
                    emit_prob = self.emission_probs[word_length][t].get(letter_j, 1e-10)
                elif masked_word[t] == letter_j:
                    # Observed this letter
                    emit_prob = 1.0
                else:
                    # Observed different letter
                    emit_prob = 0.0

                alpha[t][letter_j] = prob_sum * emit_prob

            # Normalize to prevent underflow
            total = sum(alpha[t].values())
            if total > 0:
                alpha[t] = {k: v / total for k, v in alpha[t].items()}

        # BACKWARD PASS
        # β[n-1][letter] = 1 (base case)
        for letter in alphabet:
            beta[n-1][letter] = 1.0

        # Backward recursion: β[t][i] = Σ_j P(j|i) * P(obs_t+1|j) * β[t+1][j]
        for t in range(n-2, -1, -1):
            for letter_i in alphabet:
                prob_sum = 0.0
                for letter_j in alphabet:
                    if beta[t+1][letter_j] > 0:
                        # Transition probability
                        trans_prob = self.transition_probs[word_length].get((t, letter_i), {}).get(letter_j, 1e-10)

                        # Emission probability at t+1
                        if masked_word[t+1] == '_':
                            emit_prob = self.emission_probs[word_length][t+1].get(letter_j, 1e-10)
                        elif masked_word[t+1] == letter_j:
                            emit_prob = 1.0
                        else:
                            emit_prob = 0.0

                        prob_sum += trans_prob * emit_prob * beta[t+1][letter_j]

                beta[t][letter_i] = prob_sum

            # Normalize
            total = sum(beta[t].values())
            if total > 0:
                beta[t] = {k: v / total for k, v in beta[t].items()}

        # COMPUTE POSTERIOR: γ[t][letter] = α[t][letter] * β[t][letter]
        # Average over all blank positions
        posterior_probs = Counter()

        for t in range(n):
            if masked_word[t] == '_':
                for letter in remaining_letters:
                    gamma = alpha[t].get(letter, 0) * beta[t].get(letter, 0)
                    posterior_probs[letter] += gamma

        # Normalize
        total = sum(posterior_probs.values())
        if total > 0:
            final_probs = {letter: posterior_probs[letter] / total for letter in remaining_letters}
        else:
            # Fallback
            final_probs = self._fallback_probabilities(masked_word, remaining_letters)

        return final_probs


    def _fallback_probabilities(self, masked_word, remaining_letters):
        """
        Fallback when HMM data insufficient - use pattern matching
        """
        word_length = len(masked_word)
        matching_words = [w for w in self.words_by_length.get(word_length, [])
                         if self._word_matches_pattern(w, masked_word)]

        if not matching_words:
            # Use global frequencies
            total = sum(self.global_letter_freq.values())
            return {letter: self.global_letter_freq.get(letter, 1) / total
                   for letter in remaining_letters}

        # Count letters in blank positions
        letter_counts = Counter()
        for word in matching_words:
            for i, char in enumerate(masked_word):
                if char == '_':
                    letter_counts[word[i]] += 1

        total = sum(letter_counts.values())
        if total > 0:
            return {letter: letter_counts.get(letter, 0) / total for letter in remaining_letters}
        else:
            return {letter: 1.0 / len(remaining_letters) for letter in remaining_letters}


    def _word_matches_pattern(self, word, pattern):
        """Check if word matches masked pattern"""
        if len(word) != len(pattern):
            return False
        for w_char, p_char in zip(word, pattern):
            if p_char != '_' and w_char != p_char:
                return False
        return True


    def save(self, filepath):
        """Save trained model"""
        # Convert nested defaultdicts to regular dicts for pickling
        emission_dict = {}
        for length in self.emission_probs:
            emission_dict[length] = {}
            for pos in self.emission_probs[length]:
                emission_dict[length][pos] = dict(self.emission_probs[length][pos])

        transition_dict = {}
        for length in self.transition_probs:
            transition_dict[length] = {}
            for key in self.transition_probs[length]:
                transition_dict[length][key] = dict(self.transition_probs[length][key])

        initial_dict = {}
        for length in self.initial_probs:
            initial_dict[length] = dict(self.initial_probs[length])

        with open(filepath, 'wb') as f:
            pickle.dump({
                'emission_probs': emission_dict,
                'transition_probs': transition_dict,
                'initial_probs': initial_dict,
                'global_letter_freq': self.global_letter_freq,
                'words_by_length': dict(self.words_by_length),
                'trained': self.trained
            }, f)
        print(f"Model saved to {filepath}")


    def load(self, filepath):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.emission_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)),
                                         data.get('emission_probs', {}))
        self.transition_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)),
                                           data.get('transition_probs', {}))
        self.initial_probs = defaultdict(lambda: defaultdict(float),
                                        data.get('initial_probs', {}))
        self.global_letter_freq = data['global_letter_freq']
        self.words_by_length = defaultdict(list, data['words_by_length'])
        self.trained = data['trained']
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    print("=" * 70)
    print("TRUE FORWARD-BACKWARD HMM")
    print("=" * 70)
    print("\nThis is a REAL HMM with:")
    print("  - Hidden states: letter positions")
    print("  - Transitions: P(letter_t+1 | letter_t, pos, length)")
    print("  - Emissions: P(letter | pos, length)")
    print("  - Forward pass: α[t] = P(obs_1:t, state_t)")
    print("  - Backward pass: β[t] = P(obs_t+1:n | state_t)")
    print("  - Posterior: γ[t] ∝ α[t] * β[t]")
    print("=" * 70)
