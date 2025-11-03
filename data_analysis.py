"""
Data Analysis and Preprocessing for Hangman Corpus
==================================================

This script analyzes the corpus for potential issues and preprocesses it.

PREPROCESSING CONCERNS FOR HANGMAN:
-----------------------------------
1. **Case sensitivity**: Should "Apple" and "apple" be treated as same?
   → Solution: Convert everything to UPPERCASE

2. **Non-alphabetic characters**: Hyphens, apostrophes, numbers
   → Solution: Remove or skip these words

3. **Stop words**: In Hangman, ALL words are valid (no stop words needed)
   → Insight: Unlike NLP tasks, we don't remove common words

4. **Frequency bias**: Very rare words might skew probabilities
   → Solution: We'll keep all words but track frequencies

5. **Word length distribution**: Some lengths might be underrepresented
   → Solution: Train separate models or weighted sampling

6. **Duplicates**: Same word appearing multiple times
   → Decision: Keep duplicates as they indicate frequency/importance
"""

import re
from collections import Counter, defaultdict


def analyze_corpus(corpus_file):
    """
    Analyze corpus for data quality issues

    Returns:
        dict: Analysis results
    """
    print("=" * 70)
    print("CORPUS ANALYSIS")
    print("=" * 70)

    with open(corpus_file, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f]

    analysis = {}

    # Basic stats
    analysis['total_words'] = len(words)
    print(f"\n1. Basic Statistics:")
    print(f"   Total lines: {len(words)}")

    # Empty lines
    empty_lines = sum(1 for w in words if not w)
    analysis['empty_lines'] = empty_lines
    print(f"   Empty lines: {empty_lines}")

    # Non-alphabetic words
    non_alpha = [w for w in words if w and not w.isalpha()]
    analysis['non_alpha_count'] = len(non_alpha)
    print(f"   Non-alphabetic words: {len(non_alpha)}")
    if non_alpha[:5]:
        print(f"   Examples: {non_alpha[:5]}")

    # Valid words
    valid_words = [w for w in words if w and w.isalpha()]
    analysis['valid_words'] = len(valid_words)
    print(f"   Valid alphabetic words: {len(valid_words)}")

    # Word length distribution
    print(f"\n2. Word Length Distribution:")
    length_dist = Counter(len(w) for w in valid_words)
    analysis['length_distribution'] = dict(length_dist)

    for length in sorted(length_dist.keys()):
        bar = '#' * (length_dist[length] // 100)
        print(f"   Length {length:2d}: {length_dist[length]:5d} words {bar}")

    print(f"\n   Shortest word: {min(len(w) for w in valid_words)} letters")
    print(f"   Longest word: {max(len(w) for w in valid_words)} letters")
    print(f"   Average length: {sum(len(w) for w in valid_words) / len(valid_words):.2f} letters")

    # Check for duplicates
    word_counts = Counter(w.upper() for w in valid_words)
    duplicates = {w: c for w, c in word_counts.items() if c > 1}
    analysis['unique_words'] = len(word_counts)
    analysis['duplicate_words'] = len(duplicates)

    print(f"\n3. Uniqueness:")
    print(f"   Unique words: {len(word_counts)}")
    print(f"   Duplicate words: {len(duplicates)}")
    if duplicates:
        top_dups = sorted(duplicates.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"   Most frequent duplicates:")
        for word, count in top_dups:
            print(f"     '{word}': {count} times")

    # Case distribution
    print(f"\n4. Case Distribution:")
    lower_case = sum(1 for w in valid_words if w.islower())
    upper_case = sum(1 for w in valid_words if w.isupper())
    title_case = sum(1 for w in valid_words if w.istitle() and len(w) > 1)
    mixed_case = len(valid_words) - lower_case - upper_case - title_case

    print(f"   Lowercase: {lower_case} ({100*lower_case/len(valid_words):.1f}%)")
    print(f"   UPPERCASE: {upper_case} ({100*upper_case/len(valid_words):.1f}%)")
    print(f"   Title Case: {title_case} ({100*title_case/len(valid_words):.1f}%)")
    print(f"   Mixed case: {mixed_case} ({100*mixed_case/len(valid_words):.1f}%)")

    # Letter frequency analysis
    print(f"\n5. Letter Frequency (across all words):")
    letter_counts = Counter()
    for word in valid_words:
        for letter in word.upper():
            letter_counts[letter] += 1

    total_letters = sum(letter_counts.values())
    for letter, count in letter_counts.most_common(10):
        pct = 100 * count / total_letters
        bar = '#' * int(pct)
        print(f"   {letter}: {pct:5.2f}% {bar}")

    analysis['letter_frequencies'] = dict(letter_counts)

    # Check for potential issues
    print(f"\n6. Potential Issues:")
    issues = []

    if empty_lines > 0:
        issues.append(f"[X] {empty_lines} empty lines found")
    else:
        print(f"   [OK] No empty lines")

    if non_alpha:
        issues.append(f"[X] {len(non_alpha)} words with non-alphabetic characters")
    else:
        print(f"   [OK] All words are alphabetic")

    if len(duplicates) > len(word_counts) * 0.1:
        issues.append(f"[!] High duplicate rate: {len(duplicates)} duplicates")
    else:
        print(f"   [OK] Low duplicate rate")

    # Check for underrepresented lengths
    avg_words_per_length = len(valid_words) / len(length_dist)
    sparse_lengths = [l for l, c in length_dist.items() if c < avg_words_per_length * 0.1]
    if sparse_lengths:
        issues.append(f"[!] Sparse word lengths: {sparse_lengths}")
    else:
        print(f"   [OK] Good distribution across word lengths")

    if issues:
        print("\n   Issues found:")
        for issue in issues:
            print(f"   {issue}")

    analysis['issues'] = issues

    return analysis


def preprocess_corpus(input_file, output_file=None):
    """
    Preprocess corpus for Hangman training

    PREPROCESSING STEPS:
    -------------------
    1. Read all words
    2. Convert to UPPERCASE (Hangman is case-insensitive)
    3. Remove non-alphabetic words
    4. Remove empty lines
    5. Keep duplicates (they indicate word importance/frequency)
    6. Save cleaned corpus

    Args:
        input_file: Path to raw corpus
        output_file: Path to save cleaned corpus (optional)

    Returns:
        list: Cleaned words
    """
    print("\n" + "=" * 70)
    print("PREPROCESSING CORPUS")
    print("=" * 70)

    with open(input_file, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f]

    print(f"\nOriginal words: {len(words)}")

    # Step 1: Remove empty lines
    words = [w for w in words if w]
    print(f"After removing empty lines: {len(words)}")

    # Step 2: Remove non-alphabetic words
    original_count = len(words)
    words = [w for w in words if w.isalpha()]
    removed = original_count - len(words)
    if removed > 0:
        print(f"After removing non-alphabetic: {len(words)} (removed {removed})")

    # Step 3: Convert to uppercase
    words = [w.upper() for w in words]
    print(f"Converted to UPPERCASE: {len(words)}")

    # Step 4: Optionally save
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for word in words:
                f.write(word + '\n')
        print(f"\n[OK] Cleaned corpus saved to: {output_file}")

    print(f"\n[OK] Preprocessing complete!")
    print(f"  Final word count: {len(words)}")
    print(f"  Unique words: {len(set(words))}")

    return words


def compare_test_corpus(corpus_file, test_file):
    """
    Compare test set with corpus to check for unseen words

    This is important because:
    - If test has words not in corpus, our HMM might struggle
    - We want to know how many OOV (out-of-vocabulary) words exist
    """
    print("\n" + "=" * 70)
    print("TEST vs CORPUS COMPARISON")
    print("=" * 70)

    with open(corpus_file, 'r', encoding='utf-8') as f:
        corpus_words = set(line.strip().upper() for line in f if line.strip().isalpha())

    with open(test_file, 'r', encoding='utf-8') as f:
        test_words = [line.strip().upper() for line in f if line.strip().isalpha()]

    print(f"\nCorpus size: {len(corpus_words)} unique words")
    print(f"Test set size: {len(test_words)} words")

    # Find OOV words
    oov_words = [w for w in test_words if w not in corpus_words]

    print(f"\nOut-of-vocabulary (OOV) words in test: {len(oov_words)}")
    print(f"OOV rate: {100 * len(oov_words) / len(test_words):.2f}%")

    if oov_words:
        print(f"\nFirst 10 OOV words: {oov_words[:10]}")
        print("\n[!] Note: Your HMM will need to handle unseen words!")
        print("   Strategy: Use position-based and bigram frequencies as fallback")
    else:
        print("\n[OK] All test words are in the corpus!")

    # Check length distribution
    corpus_lengths = Counter(len(w) for w in corpus_words)
    test_lengths = Counter(len(w) for w in test_words)

    print(f"\nLength distribution comparison:")
    all_lengths = sorted(set(corpus_lengths.keys()) | set(test_lengths.keys()))
    for length in all_lengths:
        corpus_count = corpus_lengths.get(length, 0)
        test_count = test_lengths.get(length, 0)
        print(f"  Length {length:2d}: Corpus={corpus_count:5d}, Test={test_count:4d}")


if __name__ == "__main__":
    # Paths
    corpus_path = r"C:\Users\laxma\OneDrive\Desktop\hmm_rl_hackathon\Data\Data\corpus.txt"
    test_path = r"C:\Users\laxma\OneDrive\Desktop\hmm_rl_hackathon\Data\Data\test.txt"
    cleaned_corpus_path = r"C:\Users\laxma\OneDrive\Desktop\hmm_rl_hackathon\corpus_cleaned.txt"

    # Run analysis
    analysis = analyze_corpus(corpus_path)

    # Preprocess
    cleaned_words = preprocess_corpus(corpus_path, cleaned_corpus_path)

    # Compare with test set
    compare_test_corpus(corpus_path, test_path)

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print("""
    Based on this analysis:

    1. [OK] Use the cleaned corpus for training (corpus_cleaned.txt)

    2. [!] Handle OOV words using:
       - Position-based letter frequencies
       - Bigram/trigram fallbacks
       - Global letter frequency as last resort

    3. [OK] Keep duplicates in training (they indicate frequency)

    4. [!] For underrepresented word lengths:
       - Train length-specific models
       - Use smoothing for sparse lengths

    5. [OK] Convert everything to UPPERCASE for consistency

    Next steps:
    -> Train your HMM on the cleaned corpus
    -> Test on partial words to verify letter predictions
    """)
