import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
import random
from collections import deque
import os 

# Constants
MAX_WORD_LENGTH = 30
HIDDEN_SIZE = 128
SWITCH_THRESHOLD = 2 # Number of missing letters to switch from RL to GPT

class HangmanEnv:
    def __init__(self, word):
        self.word = word.lower()
        self.max_lives = 6
        self.reset()

    def reset(self):
        self.lives = self.max_lives
        self.guessed_letters = set()
        self.word_state = ['_' for _ in self.word]
        return self._get_state()

    def step(self, action):
        letter = chr(action + 97)
        reward = 0
        done = False

        if letter in self.guessed_letters:
            self.lives -= 1
            reward -= 1
        else:
            self.guessed_letters.add(letter)
            if letter in self.word:
                for i, char in enumerate(self.word):
                    if char == letter:
                        self.word_state[i] = letter
                        reward += 1
            else:
                self.lives -= 1
                reward -= 0.5

        if '_' not in self.word_state:
            reward += 5
            done = True
        elif self.lives == 0:
            reward -= 3
            done = True

        return self._get_state(), reward, done

    def _get_state(self):
        word_state = [0 if c == '_' else ord(c) - 96 for c in self.word_state] + [0] * (MAX_WORD_LENGTH - len(self.word))
        guessed = [1 if chr(i + 97) in self.guessed_letters else 0 for i in range(26)]
        return word_state + guessed + [self.lives]

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size * 2)
        self.ln1 = nn.LayerNorm(hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.ln2 = nn.LayerNorm(hidden_size * 2)
        self.fc3 = nn.Linear(hidden_size * 2, hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.ln3(self.fc3(x)))
        x = self.fc4(x)
        return x

class HangmanLLM:
    def __init__(self, config=None):
        if config is None:
            config = GPT2Config(
                vocab_size=28,
                n_positions=32,
                n_ctx=32,
                n_embd=128,
                n_layer=4,
                n_head=4
            )
        self.model = GPT2LMHeadModel(config)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<MASK>']})
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.resize_token_embeddings(len(self.tokenizer))

    def predict_next_letter(self, partial_word):
        input_ids = self.tokenizer.encode(partial_word, return_tensors='pt', padding='max_length', max_length=32)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).float()
        with torch.no_grad():
            output = self.model.generate(input_ids, attention_mask=attention_mask, max_length=len(partial_word)+1, num_return_sequences=1)
        predicted_word = self.tokenizer.decode(output[0])
        for char in predicted_word:
            if char not in partial_word and char.isalpha():
                return ord(char.lower()) - 97
        return None

class HybridHangmanModel:
    def __init__(self, rl_model_path, gpt_model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load RL model
        input_size = MAX_WORD_LENGTH + 26 + 1
        self.rl_model = DQN(input_size, HIDDEN_SIZE, 26).to(self.device)
        if os.path.exists(rl_model_path):
            print(f"Loading RL model from {rl_model_path}")
            checkpoint = torch.load(rl_model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.rl_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                print("Warning: 'model_state_dict' not found in checkpoint. Attempting to load entire checkpoint.")
                self.rl_model.load_state_dict(checkpoint)
            print(f"RL model loaded successfully. Checkpoint info: Episode {checkpoint.get('episode', 'N/A')}, "
                  f"Validation Reward: {checkpoint.get('val_reward', 'N/A')}")
        else:
            print(f"RL model file {rl_model_path} not found. Initialize with random weights.")
        self.rl_model.eval()

        # Load GPT model
        self.gpt_model = HangmanLLM()
        if os.path.exists(gpt_model_path):
            print(f"Loading GPT model from {gpt_model_path}")
            self.gpt_model.model.load_state_dict(torch.load(gpt_model_path, map_location=self.device))
        else:
            print(f"GPT model file {gpt_model_path} not found. Initialize with default weights.")
        self.gpt_model.model.eval()

    def play(self, word):
        env = HangmanEnv(word)
        state = env.reset()
        done = False
        
        while not done:
            missing_letters = state.count(0)
            
            if missing_letters > SWITCH_THRESHOLD:
                # Use RL model
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = self.rl_model(state_tensor).max(1)[1].item()
            else:
                # Use GPT model
                partial_word = ''.join([chr(s + 96) if s > 0 else '_' for s in state[:len(word)]])
                action = self.gpt_model.predict_next_letter(partial_word)
                if action is None:
                    # If GPT can't predict, fall back to RL
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        action = self.rl_model(state_tensor).max(1)[1].item()

            state, reward, done = env.step(action)
            
            #print(f"Guessed: {chr(action + 97)}, Word: {''.join(env.word_state)}, Lives: {env.lives}")
        
        return ''.join(env.word_state) == word



if __name__ == "__main__":
    rl_model_path = "RL.pth"
    gpt_model_path = "GPT.pth"
    
    hybrid_model = HybridHangmanModel(rl_model_path, gpt_model_path)
    """
    # Test on predefined words
    test_words = ['python', 'hangman', 'artificial', 'intelligence', 'donaldduck', 'exeter']
    for word in test_words:
        print(f"\nPlaying Hangman with word: {word}")
        result = hybrid_model.play(word)
        print(f"Result: {'Won' if result else 'Lost'}")"""

    # Test on 10 random words from the original dictionary
    print("\n--- Testing on 10 random words from the original dictionary ---")
    
    # Load words from the dictionary file
    with open('words_250000_train.txt', 'r') as f:
        dictionary_words = f.read().splitlines()
    
    # Select 10 random words
    random_test_words = random.sample(dictionary_words, 100000)
    
    wins = 0
    for word in random_test_words:
        #print(f"\nPlaying Hangman with word: {word}")
        result = hybrid_model.play(word)
        if result:
            wins += 1
        #print(f"Result: {'Won' if result else 'Lost'}")
    
    print(f"\nFinal Result: Won {wins} out of 100k games with random words.")
    print(f"Win Rate: {wins/100000 * 100:.2f}%")
