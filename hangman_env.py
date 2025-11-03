"""
Hangman Game Environment
========================

This is the environment where the RL agent will play Hangman.

ENVIRONMENT COMPONENTS (Standard RL):
-------------------------------------
1. STATE: Current game situation (masked word, guessed letters, lives left)
2. ACTION: Guessing a letter
3. REWARD: Feedback for the action (good guess = positive, bad guess = negative)
4. DONE: Whether game is over (won or lost)

GAME RULES:
-----------
- Player has 6 wrong guesses allowed
- Guessing a correct letter reveals all occurrences
- Guessing an incorrect letter costs 1 life
- Guessing a repeated letter is penalized
- Win: Reveal full word before running out of lives
- Lose: Run out of lives before revealing word
"""

import random
import string
from typing import Set, Tuple, Dict


class HangmanEnv:
    """
    Hangman game environment for RL training

    This follows the Gymnasium/OpenAI Gym interface pattern
    """

    def __init__(self, word_list, max_wrong_guesses=6):
        """
        Initialize Hangman environment

        Args:
            word_list: List of words to use for game
            max_wrong_guesses: Maximum number of wrong guesses allowed (default: 6)
        """
        self.word_list = [w.upper() for w in word_list]
        self.max_wrong_guesses = max_wrong_guesses

        # Current game state
        self.target_word = None
        self.masked_word = None
        self.guessed_letters = set()
        self.wrong_guesses = 0
        self.done = False

        # Statistics
        self.total_games = 0
        self.total_wins = 0
        self.total_wrong_guesses = 0
        self.total_repeated_guesses = 0


    def reset(self, word=None):
        """
        Reset environment to start a new game

        Args:
            word: Optional specific word to use (for testing)

        Returns:
            state: Initial game state
        """
        # Choose a random word
        if word is None:
            self.target_word = random.choice(self.word_list)
        else:
            self.target_word = word.upper()

        # Initialize game state
        self.masked_word = '_' * len(self.target_word)
        self.guessed_letters = set()
        self.wrong_guesses = 0
        self.done = False

        return self.get_state()


    def step(self, action):
        """
        Take an action (guess a letter) and return result

        This is the CORE of the RL environment!

        Args:
            action: Letter to guess (string, e.g., 'A')

        Returns:
            state: New game state after action
            reward: Reward for this action
            done: Whether game is over
            info: Additional information (for debugging)
        """
        if self.done:
            raise ValueError("Game is already over! Call reset() to start new game.")

        action = action.upper()

        # Check if letter was already guessed (REPEATED GUESS)
        if action in self.guessed_letters:
            reward = -50  # Heavy penalty for inefficiency
            self.total_repeated_guesses += 1
            info = {
                'result': 'repeated',
                'message': f"Letter '{action}' was already guessed!"
            }
            return self.get_state(), reward, self.done, info

        # Add to guessed letters
        self.guessed_letters.add(action)

        # Check if letter is in word (CORRECT GUESS)
        if action in self.target_word:
            # Reveal all occurrences of this letter
            new_masked = list(self.masked_word)
            occurrences = 0

            for i, letter in enumerate(self.target_word):
                if letter == action:
                    new_masked[i] = action
                    occurrences += 1

            self.masked_word = ''.join(new_masked)

            # Reward: more for revealing multiple letters
            reward = 10 + (5 * occurrences)

            # Check if won
            if '_' not in self.masked_word:
                reward += 100  # Big bonus for winning!
                self.done = True
                self.total_wins += 1
                info = {
                    'result': 'win',
                    'message': f"Won! Word was '{self.target_word}'"
                }
            else:
                info = {
                    'result': 'correct',
                    'message': f"Correct! '{action}' appears {occurrences} time(s)"
                }

        # WRONG GUESS
        else:
            self.wrong_guesses += 1
            self.total_wrong_guesses += 1
            reward = -20  # Penalty for wrong guess

            # Check if lost
            if self.wrong_guesses >= self.max_wrong_guesses:
                reward -= 50  # Extra penalty for losing
                self.done = True
                info = {
                    'result': 'loss',
                    'message': f"Lost! Word was '{self.target_word}'"
                }
            else:
                info = {
                    'result': 'wrong',
                    'message': f"Wrong! '{action}' not in word. Lives left: {self.max_wrong_guesses - self.wrong_guesses}"
                }

        # Update statistics
        if self.done:
            self.total_games += 1

        return self.get_state(), reward, self.done, info


    def get_state(self):
        """
        Get current game state

        STATE REPRESENTATION:
        --------------------
        This is what the RL agent "sees"

        Returns:
            dict: Current state information
        """
        return {
            'masked_word': self.masked_word,
            'guessed_letters': self.guessed_letters.copy(),
            'wrong_guesses': self.wrong_guesses,
            'lives_left': self.max_wrong_guesses - self.wrong_guesses,
            'word_length': len(self.target_word),
            'letters_revealed': sum(1 for c in self.masked_word if c != '_'),
            'done': self.done
        }


    def get_available_actions(self):
        """
        Get list of letters that haven't been guessed yet

        Returns:
            set: Unguessed letters
        """
        all_letters = set(string.ascii_uppercase)
        return all_letters - self.guessed_letters


    def render(self, mode='human'):
        """
        Display current game state (for debugging/visualization)

        Args:
            mode: 'human' for text display
        """
        print("\n" + "=" * 50)
        print(f"Word: {self.masked_word}")
        print(f"Guessed: {sorted(self.guessed_letters)}")
        print(f"Wrong guesses: {self.wrong_guesses}/{self.max_wrong_guesses}")
        print(f"Lives left: {self.max_wrong_guesses - self.wrong_guesses}")

        # Draw hangman (traditional visualization)
        hangman_stages = [
            """
               ------
               |    |
               |
               |
               |
               |
            --------
            """,
            """
               ------
               |    |
               |    O
               |
               |
               |
            --------
            """,
            """
               ------
               |    |
               |    O
               |    |
               |
               |
            --------
            """,
            """
               ------
               |    |
               |    O
               |   /|
               |
               |
            --------
            """,
            """
               ------
               |    |
               |    O
               |   /|\\
               |
               |
            --------
            """,
            """
               ------
               |    |
               |    O
               |   /|\\
               |   /
               |
            --------
            """,
            """
               ------
               |    |
               |    O
               |   /|\\
               |   / \\
               |
            --------
            """
        ]

        if self.wrong_guesses < len(hangman_stages):
            print(hangman_stages[self.wrong_guesses])

        print("=" * 50)


    def get_statistics(self):
        """
        Get overall game statistics

        Returns:
            dict: Statistics
        """
        if self.total_games == 0:
            return {
                'total_games': 0,
                'win_rate': 0.0,
                'avg_wrong_guesses': 0.0,
                'total_repeated_guesses': 0
            }

        return {
            'total_games': self.total_games,
            'total_wins': self.total_wins,
            'win_rate': self.total_wins / self.total_games,
            'avg_wrong_guesses': self.total_wrong_guesses / self.total_games,
            'total_repeated_guesses': self.total_repeated_guesses
        }


def play_interactive_game(env):
    """
    Play an interactive game (for testing the environment)

    Args:
        env: HangmanEnv instance
    """
    print("\n" + "=" * 60)
    print("INTERACTIVE HANGMAN GAME")
    print("=" * 60)
    print("Type a letter to guess, or 'quit' to exit")

    state = env.reset()
    env.render()

    while not state['done']:
        # Get user input
        guess = input("\nYour guess: ").strip().upper()

        if guess == 'QUIT':
            print("Quitting...")
            break

        if len(guess) != 1 or guess not in string.ascii_uppercase:
            print("Invalid input! Please enter a single letter.")
            continue

        if guess in state['guessed_letters']:
            print(f"You already guessed '{guess}'!")
            continue

        # Take action
        state, reward, done, info = env.step(guess)

        # Show result
        print(f"\n{info['message']}")
        print(f"Reward: {reward}")

        env.render()

    # Show final result
    if state['done']:
        if env.wrong_guesses >= env.max_wrong_guesses:
            print(f"\n GAME OVER! The word was: {env.target_word}")
        else:
            print(f"\n CONGRATULATIONS! You won!")


# Example usage and testing
if __name__ == "__main__":
    # Load some test words
    test_words = ["APPLE", "BANANA", "CHERRY", "DRAGON", "ELEPHANT"]

    # Create environment
    env = HangmanEnv(test_words)

    # Test 1: Play a scripted game
    print("=" * 60)
    print("TEST 1: Scripted Game")
    print("=" * 60)

    state = env.reset(word="APPLE")
    print(f"\nTarget word: {env.target_word} (for testing)")
    env.render()

    # Simulate good guesses
    for letter in ['E', 'A', 'P', 'L']:
        print(f"\nGuessing: {letter}")
        state, reward, done, info = env.step(letter)
        print(f"Result: {info['message']}")
        print(f"Reward: {reward}")
        env.render()

        if done:
            break

    # Test 2: Play interactive game (uncomment to play)
    # play_interactive_game(env)

    print("\n\nEnvironment is ready for RL training!")
