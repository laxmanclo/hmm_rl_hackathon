# HMM Training Process Explained

## What Happens During Training

### Input: 49,979 words from corpus_cleaned.txt

### The HMM learns THREE types of patterns:

---

## 1. POSITIONAL FREQUENCIES

For each word length and each position, count which letters appear there.

**Example**: Training on "APPLE", "APPLY", "AMPLE"

```
Word length = 5:

Position 0 (first letter):
  A: 3 times (APPLE, APPLY, AMPLE)

Position 1 (second letter):
  P: 2 times (APPLE, APPLY)
  M: 1 time  (AMPLE)

Position 2 (third letter):
  P: 2 times (APPLE, APPLY)
  P: 1 time  (AMPLE)

Position 3 (fourth letter):
  L: 3 times (all words)

Position 4 (fifth letter):
  E: 2 times (APPLE, AMPLE)
  Y: 1 time  (APPLY)
```

**During Hangman**: If we see `_PP__`, HMM knows:
- Position 0: A is very likely (appeared often at start)
- Position 3: L is very likely (appeared often after PP)
- Position 4: E or Y are likely

---

## 2. BIGRAM TRANSITIONS

Count which letters follow which letters (pairs of consecutive letters).

**Example**: Training on "APPLE", "APPLY"

```
Bigrams extracted:
A-P (from APPLE, APPLY)
P-P (from APPLE, APPLY)
P-L (from APPLE, APPLY)
L-E (from APPLE)
L-Y (from APPLY)

Bigram frequencies:
After 'A': P appears 2 times
After 'P': P appears 2 times, L appears 2 times
After 'L': E appears 1 time, Y appears 1 time
```

**During Hangman**: If we see `_PP__`:
- We know position 1 is 'P'
- HMM checks: "After P, what usually comes?"
- Answer: P (very common), L (very common)
- This suggests next letter might be L

---

## 3. GLOBAL LETTER FREQUENCY

Overall, which letters appear most in English?

From our corpus:
```
E: 10.37% (most common)
A:  8.87%
I:  8.86%
O:  7.54%
...
Z:  0.15% (rare)
```

**During Hangman**: If HMM has no pattern match (rare word length or unusual pattern):
- Fall back to: "Just guess the most common letters"
- This is the safety net!

---

## How These Combine During Prediction

### Example: masked_word = "_PP_E", guessed = {P, E}

```python
# Step 1: Pattern Matching
matching_words = find_words_matching("_PP_E")
# Might find: APPLE (if in corpus)
# Extract: position 0 could be A, position 3 could be L

# Step 2: Positional Probabilities
# Look at position 0 in 5-letter words: A appears X%, B appears Y%
# Look at position 3 in 5-letter words: L appears X%, ...

# Step 3: Bigram Context
# We see "P" at position 1
# What comes after "P"? → P (position 2), then L (likely)

# Step 4: Combine with weights
for each remaining letter:
    score = 0.5 * pattern_match_prob +
            0.3 * positional_prob +
            0.2 * bigram_prob

# Step 5: Normalize and return
# L might get score 0.35
# A might get score 0.25
# ...return sorted by score
```

---

## Training Data Structures

After training, HMM stores:

```python
letter_freq_by_length = {
    5: {  # word length 5
        0: {'A': 523, 'B': 89, 'C': 234, ...},  # position 0 counts
        1: {'A': 412, 'B': 102, ...},            # position 1 counts
        ...
    },
    6: {...},
    7: {...},
    ...
}

bigram_freq = {
    'A': {'B': 45, 'C': 89, 'D': 12, ...},  # after A, B appears 45 times
    'B': {'A': 67, 'E': 123, ...},
    ...
}

global_letter_freq = {
    'A': 45823,  # total count across all words
    'B': 12456,
    ...
}

words_by_length = {
    5: ['APPLE', 'APPLY', 'AMPLE', ...],
    6: [...],
    ...
}
```

---

## Why This Works for 100% OOV Test Set

Even though test words are ALL unseen:

1. **Patterns generalize**:
   - Learned: "PP" is common
   - Test has "HOPPING" (unseen word, but HMM knows PP pattern)

2. **Position frequencies generalize**:
   - Learned: Position 0 in 5-letter words often has A, S, T
   - Test word starts with S? HMM predicts well!

3. **Bigrams generalize**:
   - Learned: After 'T', 'H' is very common
   - Test has "THROW"? HMM suggests H after T!

This is the power of statistical learning vs. memorization!
