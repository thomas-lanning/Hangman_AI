# Hangman_AI
Many iterations for a hangman AI which I trained for a coding interview

It starts off for 2 letter using reinforcement learning and then uses a GPT to guess the remaining letters.
I used a GPT instance to try guess the hangman words to try something different...
Cannot include the model path files nor the word list since they want it to be kept confidential.

Here is a short summary:

1. Hangman Environment (HangmanEnv)
  This class sets up the Hangman game environment. It takes a word, initializes the number of lives, and tracks guessed letters. It then creates a word state with underscores representing unguessed letters.
The step() function takes an action (a guessed letter) and updates the game:
If the letter has already been guessed, it reduces the number of lives.
If the letter is in the word, the word state is updated.
The game ends if the word is fully guessed or if I run out of lives.
3. Reinforcement Learning Model (DQN)
  I use a Deep Q-Network (DQN) to predict the next letter based on the current state of the game.
The model has four layers, including dropout and layer normalization, to process the input, which includes the current word state, guessed letters, and remaining lives.
In the forward() function, the model outputs Q-values for each possible action (letter to guess).
4. GPT-Based Language Model (HangmanLLM)
  I also incorporate a GPT-based model for predicting the next letter. It uses the GPT-2 architecture, but I’ve customized it to handle Hangman.
The model uses the predict_next_letter() function, which takes a partially completed word and generates the most likely next letter. If the model can’t predict a letter, it falls back to the RL model.
5. Hybrid Hangman Model (HybridHangmanModel)
  The hybrid model combines both the RL and GPT models. It switches between the two based on the number of missing letters in the word:
If there are too many missing letters, it switches to the RL model.
If there are fewer missing letters, it uses the GPT model to predict the next letter.
I load both models (RL and GPT) from saved checkpoints, or initialize them with random weights if no saved model is available.
6. Testing
  I test the hybrid model by playing Hangman with both predefined words and randomly selected words from a dictionary. The model's performance is evaluated by the number of games won out of 100,000 randomly chosen words.
