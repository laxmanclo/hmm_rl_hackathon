"""
Hidden Markov Model for Hangman
================================

This module implements an HMM-based letter predictor for Hangman.

CONCEPT EXPLANATION:
-------------------
An HMM learns patterns in words. For Hangman, we use it to answer:
"Given partial word _PP_E and position, what letter is likely?"

COMPONENTS:
-----------
1. States: Positions in words (0, 1, 2, ...)
2. Observations: Letters (A-Z)
3. Transitions: P(letter at position i+1 | letter at position i)
4. Emissions: P(letter L at position i)

MATHEMATICS:
-----------
For a word like "APPLE" with pattern "_PP_E":
- We want: P(letter | pattern, position)
- We compute: frequency of letters appearing at each position
- Given length-based context

"""

import numpy as np
from collections import defaultdict, Counter
import pickle
import string


class HangmanHMM:
    """
    HMM-inspired letter probability predictor

    This is a simplified HMM that uses:
    - Position-based letter frequencies
    - Pattern matching with known words
    - N-gram context (letter sequences)
    """

    def __init__(self):
        # Store frequency data by word length
        self.letter_freq_by_length = defaultdict(lambda: defaultdict(Counter))
        # Format: {length: {position: Counter({letter: count})}}

        # Store bigram transitions (letter pairs)
        self.bigram_freq = defaultdict(Counter)
        # Format: {letter: Counter({next_letter: count})}

        # NEW: Store trigram transitions (3-letter sequences)
        self.trigram_freq = defaultdict(lambda: defaultdict(Counter))
        # Format: {(letter1, letter2): Counter({next_letter: count})}

        # NEW: Store 4-gram transitions (4-letter sequences)
        # Using tuple keys to avoid nested lambda pickle issues
        self.fourgram_freq = defaultdict(Counter)
        # Format: {(letter1, letter2, letter3): Counter({next_letter: count})}

        # NEW: Store 5-gram patterns (for very common endings)
        self.fivegram_freq = defaultdict(Counter)
        # Format: {(4_letter_prefix): Counter({next_letter: count})}

        # NEW: Store common suffixes and prefixes
        self.suffix_freq = defaultdict(Counter)  # {suffix: Counter({letter_before: count})}
        self.prefix_freq = defaultdict(Counter)  # {prefix: Counter({letter_after: count})}

        # Store overall letter frequencies
        self.global_letter_freq = Counter()

        # Store all words by length for pattern matching
        self.words_by_length = defaultdict(list)

        self.trained = False


    def train(self, corpus_file):
        """
        Train the HMM on a corpus of words

        TRAINING PROCESS:
        ----------------
        1. Read all words from corpus
        2. For each word:
           a. Count letter at each position (for positional probabilities)
           b. Count bigrams (letter pairs) for transition probabilities
           c. Count overall letter frequencies
        3. Store words grouped by length for pattern matching

        Args:
            corpus_file: Path to text file with one word per line
        """
        print("Training HMM on corpus...")
        word_count = 0

        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().upper()

                # Skip invalid words
                if not word or not word.isalpha():
                    continue

                word_count += 1
                word_length = len(word)

                # Store word for pattern matching
                self.words_by_length[word_length].append(word)

                # Count letter frequencies at each position
                for pos, letter in enumerate(word):
                    self.letter_freq_by_length[word_length][pos][letter] += 1
                    self.global_letter_freq[letter] += 1

                # Count bigrams (consecutive letter pairs)
                for i in range(len(word) - 1):
                    current_letter = word[i]
                    next_letter = word[i + 1]
                    self.bigram_freq[current_letter][next_letter] += 1

                # NEW: Count trigrams (3-letter sequences)
                for i in range(len(word) - 2):
                    letter1 = word[i]
                    letter2 = word[i + 1]
                    next_letter = word[i + 2]
                    self.trigram_freq[letter1][letter2][next_letter] += 1

                # NEW: Count 4-grams (4-letter sequences)
                for i in range(len(word) - 3):
                    trigram = (word[i], word[i + 1], word[i + 2])
                    next_letter = word[i + 3]
                    self.fourgram_freq[trigram][next_letter] += 1

                # NEW: Count 5-grams (for very specific patterns like -ATION, -ITION)
                if word_length >= 5:
                    for i in range(len(word) - 4):
                        fourgram_prefix = word[i:i+4]
                        next_letter = word[i + 4]
                        self.fivegram_freq[fourgram_prefix][next_letter] += 1

                # NEW: Extract common suffixes (last 2-4 letters)
                if word_length >= 3:
                    for suffix_len in [2, 3, 4]:
                        if word_length > suffix_len:
                            suffix = word[-suffix_len:]
                            letter_before = word[-(suffix_len + 1)]
                            self.suffix_freq[suffix][letter_before] += 1

                # NEW: Extract common prefixes (first 2-4 letters)
                if word_length >= 3:
                    for prefix_len in [2, 3, 4]:
                        if word_length > prefix_len:
                            prefix = word[:prefix_len]
                            letter_after = word[prefix_len]
                            self.prefix_freq[prefix][letter_after] += 1

                if word_count % 10000 == 0:
                    print(f"  Processed {word_count} words...")

        self.trained = True
        print(f"Training complete! Processed {word_count} words")
        print(f"Word length range: {min(self.words_by_length.keys())} to {max(self.words_by_length.keys())}")


    def get_letter_probabilities(self, masked_word, guessed_letters):
        """
        Get probability distribution for next letter to guess

        ALGORITHM:
        ---------
        1. Filter words matching the pattern (e.g., _PP_E matches APPLE)
        2. Count letters in blank positions
        3. Combine with positional frequencies and bigram context
        4. Return normalized probabilities

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

        # Strategy 1: Pattern matching
        matching_words = self._get_matching_words(masked_word, word_length)
        pattern_probs = self._get_pattern_probabilities(matching_words, masked_word, remaining_letters)

        # Strategy 2: Position-based frequencies
        position_probs = self._get_positional_probabilities(masked_word, word_length, remaining_letters)

        # Strategy 3: Bigram context
        bigram_probs = self._get_bigram_probabilities(masked_word, remaining_letters)

        # NEW: Strategy 4: Trigram context (stronger than bigrams)
        trigram_probs = self._get_trigram_probabilities(masked_word, remaining_letters)

        # NEW: Strategy 5: 4-gram context (very specific, strong signal)
        fourgram_probs = self._get_fourgram_probabilities(masked_word, remaining_letters)

        # NEW: Strategy 6: 5-gram context (extremely specific)
        fivegram_probs = self._get_fivegram_probabilities(masked_word, remaining_letters)

        # NEW: Strategy 7: Suffix/prefix patterns
        suffix_probs = self._get_suffix_probabilities(masked_word, remaining_letters)
        prefix_probs = self._get_prefix_probabilities(masked_word, remaining_letters)

        # Combine strategies with weights (OPTIMIZED for OOV generalization)
        combined_probs = {}
        for letter in remaining_letters:
            # OPTIMIZED WEIGHTS: Strongly favor longer n-grams
            # Longer n-grams = more context = better predictions on unseen words
            combined_probs[letter] = (
                0.15 * pattern_probs.get(letter, 0) +     # Pattern matching (corpus-specific)
                0.03 * position_probs.get(letter, 0) +    # Position freq (too general)
                0.02 * bigram_probs.get(letter, 0) +      # Bigrams (too short)
                0.25 * trigram_probs.get(letter, 0) +     # Trigrams (good balance)
                0.20 * fourgram_probs.get(letter, 0) +    # 4-grams (very specific)
                0.15 * fivegram_probs.get(letter, 0) +    # 5-grams (extremely specific)
                0.12 * suffix_probs.get(letter, 0) +      # Suffix patterns
                0.08 * prefix_probs.get(letter, 0)        # Prefix patterns
            )

        # Normalize to sum to 1.0
        total = sum(combined_probs.values())
        if total > 0:
            combined_probs = {k: v / total for k, v in combined_probs.items()}
        else:
            # Fallback to uniform distribution
            combined_probs = {k: 1.0 / len(remaining_letters) for k in remaining_letters}

        return combined_probs


    def _get_matching_words(self, masked_word, word_length):
        """
        Find all words matching the pattern

        Example: _PP_E matches APPLE, but not AMPLE
        """
        candidates = self.words_by_length.get(word_length, [])
        matching = []

        for word in candidates:
            if self._word_matches_pattern(word, masked_word):
                matching.append(word)

        return matching


    def _word_matches_pattern(self, word, pattern):
        """
        Check if word matches the masked pattern

        Args:
            word: Full word, e.g., "APPLE"
            pattern: Masked pattern, e.g., "_PP_E"

        Returns:
            bool: True if word fits pattern
        """
        if len(word) != len(pattern):
            return False

        for w_char, p_char in zip(word, pattern):
            if p_char != '_' and w_char != p_char:
                return False

        return True


    def _get_pattern_probabilities(self, matching_words, masked_word, remaining_letters):
        """
        Get letter probabilities from pattern-matching words
        """
        if not matching_words:
            return {letter: 0 for letter in remaining_letters}

        letter_counts = Counter()

        # Count letters appearing in blank positions
        for word in matching_words:
            for i, char in enumerate(masked_word):
                if char == '_':
                    letter_counts[word[i]] += 1

        # Normalize
        total = sum(letter_counts.values())
        if total == 0:
            return {letter: 0 for letter in remaining_letters}

        probs = {letter: letter_counts.get(letter, 0) / total for letter in remaining_letters}
        return probs


    def _get_positional_probabilities(self, masked_word, word_length, remaining_letters):
        """
        Get letter probabilities based on position frequencies
        """
        position_counts = Counter()

        # Sum frequencies across all blank positions
        for pos, char in enumerate(masked_word):
            if char == '_':
                pos_freq = self.letter_freq_by_length[word_length].get(pos, Counter())
                for letter in remaining_letters:
                    position_counts[letter] += pos_freq.get(letter, 0)

        # Normalize
        total = sum(position_counts.values())
        if total == 0:
            return {letter: 0 for letter in remaining_letters}

        probs = {letter: position_counts.get(letter, 0) / total for letter in remaining_letters}
        return probs


    def _get_bigram_probabilities(self, masked_word, remaining_letters):
        """
        Get letter probabilities based on bigram context

        Example: In _PP_, if we see 'P', what comes after 'P'?
        """
        bigram_counts = Counter()

        # Look at revealed letters and their context
        for i in range(len(masked_word) - 1):
            if masked_word[i] != '_' and masked_word[i + 1] == '_':
                # We know current letter, next is blank
                current_letter = masked_word[i]
                next_letter_freq = self.bigram_freq.get(current_letter, Counter())
                for letter in remaining_letters:
                    bigram_counts[letter] += next_letter_freq.get(letter, 0)

        # Normalize
        total = sum(bigram_counts.values())
        if total == 0:
            # Fallback to global frequencies
            total = sum(self.global_letter_freq.values())
            probs = {letter: self.global_letter_freq.get(letter, 0) / total
                    for letter in remaining_letters}
        else:
            probs = {letter: bigram_counts.get(letter, 0) / total
                    for letter in remaining_letters}

        return probs


    def _get_trigram_probabilities(self, masked_word, remaining_letters):
        """
        Get letter probabilities based on trigram context

        Example: In _TION, if we see 'TI', what comes after?
        Trigrams are more specific than bigrams and capture longer patterns.
        """
        trigram_counts = Counter()

        # Look for two consecutive revealed letters before a blank
        for i in range(len(masked_word) - 2):
            if (masked_word[i] != '_' and
                masked_word[i + 1] != '_' and
                masked_word[i + 2] == '_'):
                # We have two known letters, next is blank
                letter1 = masked_word[i]
                letter2 = masked_word[i + 1]

                # Get trigram frequencies
                if letter1 in self.trigram_freq and letter2 in self.trigram_freq[letter1]:
                    next_letter_freq = self.trigram_freq[letter1][letter2]
                    for letter in remaining_letters:
                        trigram_counts[letter] += next_letter_freq.get(letter, 0)

        # Normalize
        total = sum(trigram_counts.values())
        if total == 0:
            return {letter: 0 for letter in remaining_letters}

        probs = {letter: trigram_counts.get(letter, 0) / total for letter in remaining_letters}
        return probs


    def _get_fourgram_probabilities(self, masked_word, remaining_letters):
        """
        Get letter probabilities based on 4-gram context

        Example: In _TION, if we see 'TIO', what comes after? → N (99%)
        4-grams are very specific and capture strong patterns.
        """
        fourgram_counts = Counter()

        # Look for three consecutive revealed letters before a blank
        for i in range(len(masked_word) - 3):
            if (masked_word[i] != '_' and
                masked_word[i + 1] != '_' and
                masked_word[i + 2] != '_' and
                masked_word[i + 3] == '_'):
                # We have three known letters, next is blank
                trigram = (masked_word[i], masked_word[i + 1], masked_word[i + 2])

                # Get 4-gram frequencies
                if trigram in self.fourgram_freq:
                    next_letter_freq = self.fourgram_freq[trigram]
                    for letter in remaining_letters:
                        fourgram_counts[letter] += next_letter_freq.get(letter, 0)

        # Normalize
        total = sum(fourgram_counts.values())
        if total == 0:
            return {letter: 0 for letter in remaining_letters}

        probs = {letter: fourgram_counts.get(letter, 0) / total for letter in remaining_letters}
        return probs


    def _get_fivegram_probabilities(self, masked_word, remaining_letters):
        """
        Get letter probabilities based on 5-gram context

        Example: In _ATION, if we see 'ATIO', what comes after? → N (100%)
        5-grams catch very specific patterns like -ATION, -ITION, -UTION.
        """
        fivegram_counts = Counter()

        # Look for four consecutive revealed letters before a blank
        for i in range(len(masked_word) - 4):
            if (masked_word[i] != '_' and
                masked_word[i + 1] != '_' and
                masked_word[i + 2] != '_' and
                masked_word[i + 3] != '_' and
                masked_word[i + 4] == '_'):
                # We have four known letters, next is blank
                fourgram_prefix = masked_word[i:i+4]

                # Get 5-gram frequencies
                if fourgram_prefix in self.fivegram_freq:
                    next_letter_freq = self.fivegram_freq[fourgram_prefix]
                    for letter in remaining_letters:
                        fivegram_counts[letter] += next_letter_freq.get(letter, 0)

        # Normalize
        total = sum(fivegram_counts.values())
        if total == 0:
            return {letter: 0 for letter in remaining_letters}

        probs = {letter: fivegram_counts.get(letter, 0) / total for letter in remaining_letters}
        return probs


    def _get_suffix_probabilities(self, masked_word, remaining_letters):
        """
        Get letter probabilities based on common suffix patterns

        Example: If word ends with 'ING', what letter comes before?
        Common suffixes: -ING, -TION, -ED, -ER, -LY, -EST, etc.
        """
        suffix_counts = Counter()

        # Check for suffixes of length 2-4 at the end
        for suffix_len in [2, 3, 4]:
            if len(masked_word) > suffix_len:
                # Get the suffix (last suffix_len characters)
                suffix = masked_word[-suffix_len:]

                # Check if suffix is fully revealed (no blanks)
                if '_' not in suffix:
                    # Check if position before suffix is blank
                    if masked_word[-(suffix_len + 1)] == '_':
                        # We know the suffix, need letter before it
                        suffix_freq = self.suffix_freq.get(suffix, Counter())
                        for letter in remaining_letters:
                            suffix_counts[letter] += suffix_freq.get(letter, 0)

        # Normalize
        total = sum(suffix_counts.values())
        if total == 0:
            return {letter: 0 for letter in remaining_letters}

        probs = {letter: suffix_counts.get(letter, 0) / total for letter in remaining_letters}
        return probs


    def _get_prefix_probabilities(self, masked_word, remaining_letters):
        """
        Get letter probabilities based on common prefix patterns

        Example: If word starts with 'UN', what letter comes after?
        Common prefixes: UN-, RE-, IN-, DIS-, etc.
        """
        prefix_counts = Counter()

        # Check for prefixes of length 2-4 at the start
        for prefix_len in [2, 3, 4]:
            if len(masked_word) > prefix_len:
                # Get the prefix (first prefix_len characters)
                prefix = masked_word[:prefix_len]

                # Check if prefix is fully revealed (no blanks)
                if '_' not in prefix:
                    # Check if position after prefix is blank
                    if masked_word[prefix_len] == '_':
                        # We know the prefix, need letter after it
                        prefix_freq = self.prefix_freq.get(prefix, Counter())
                        for letter in remaining_letters:
                            prefix_counts[letter] += prefix_freq.get(letter, 0)

        # Normalize
        total = sum(prefix_counts.values())
        if total == 0:
            return {letter: 0 for letter in remaining_letters}

        probs = {letter: prefix_counts.get(letter, 0) / total for letter in remaining_letters}
        return probs


    def save(self, filepath):
        """Save trained model to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'letter_freq_by_length': dict(self.letter_freq_by_length),
                'bigram_freq': dict(self.bigram_freq),
                'trigram_freq': dict(self.trigram_freq),
                'fourgram_freq': dict(self.fourgram_freq),
                'fivegram_freq': dict(self.fivegram_freq),
                'suffix_freq': dict(self.suffix_freq),
                'prefix_freq': dict(self.prefix_freq),
                'global_letter_freq': self.global_letter_freq,
                'words_by_length': dict(self.words_by_length),
                'trained': self.trained
            }, f)
        print(f"Model saved to {filepath}")


    def load(self, filepath):
        """Load trained model from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.letter_freq_by_length = defaultdict(lambda: defaultdict(Counter), data['letter_freq_by_length'])
        self.bigram_freq = defaultdict(Counter, data['bigram_freq'])

        # Load new features with fallback for old models
        self.trigram_freq = defaultdict(lambda: defaultdict(Counter),
                                       data.get('trigram_freq', {}))
        self.fourgram_freq = defaultdict(Counter, data.get('fourgram_freq', {}))
        self.fivegram_freq = defaultdict(Counter, data.get('fivegram_freq', {}))
        self.suffix_freq = defaultdict(Counter, data.get('suffix_freq', {}))
        self.prefix_freq = defaultdict(Counter, data.get('prefix_freq', {}))

        self.global_letter_freq = data['global_letter_freq']
        self.words_by_length = defaultdict(list, data['words_by_length'])
        self.trained = data['trained']
        print(f"Model loaded from {filepath}")


# Example usage and testing
if __name__ == "__main__":
    # Test with a small corpus
    print("=" * 60)
    print("HMM Model Test")
    print("=" * 60)

    # You would call this with your actual corpus:
    # hmm = HangmanHMM()
    # hmm.train('corpus.txt')
    # hmm.save('hmm_model.pkl')

    print("\nTo use this HMM:")
    print("1. hmm = HangmanHMM()")
    print("2. hmm.train('corpus.txt')")
    print("3. probs = hmm.get_letter_probabilities('_PP_E', {'P', 'E', 'S'})")
    print("4. print(sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5])")
