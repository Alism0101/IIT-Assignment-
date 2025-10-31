import torch
import torch.nn as nn
import torch.optim as optim
from model import WordMuncher, WordBaker, TranslatorBot9000
import numpy as np

HOW_MANY_IN_WORDS = 100
HOW_MANY_OUT_WORDS = 120
SPARKLE_FACTOR = 128
BRAIN_SIZE = 256
HOW_MANY_LAYERS = 1
BRAIN_CELL_TYPE = 'GRU'

muncher = WordMuncher(HOW_MANY_IN_WORDS, SPARKLE_FACTOR, BRAIN_SIZE, HOW_MANY_LAYERS, BRAIN_CELL_TYPE)
baker = WordBaker(HOW_MANY_OUT_WORDS, SPARKLE_FACTOR, BRAIN_SIZE, HOW_MANY_LAYERS, BRAIN_CELL_TYPE)
bot = TranslatorBot9000(muncher, baker)

learningJuice = optim.Adam(bot.parameters())
sadnessCalculator = nn.NLLLoss(ignore_index=0)

def oneRoundOfSchool(da_bot, da_juice, da_calc):
    da_bot.train()
    totalSadness = 0
    
    in_words = torch.randint(1, HOW_MANY_IN_WORDS, (32, 10))
    out_words = torch.randint(1, HOW_MANY_OUT_WORDS, (32, 10))

    da_juice.zero_grad()
    
    allTheGuesses = da_bot(in_words, out_words)
    
    guess_dim = allTheGuesses.shape[-1]
    
    allTheGuesses = allTheGuesses[:, 1:].reshape(-1, guess_dim)
    out_words = out_words[:, 1:].reshape(-1)
    
    currentSadness = da_calc(allTheGuesses, out_words)
    
    currentSadness.backward()
    
    da_juice.step()
    
    return currentSadness.item()

if __name__ == "__main__":
    print("Sending bot to school...")
    sadnessLevel = oneRoundOfSchool(bot, learningJuice, sadnessCalculator)
    print(f"Sadness level after one class: {sadnessLevel}")
    
    if np.isnan(sadnessLevel) or np.isinf(sadnessLevel):
        print("\nBot's brain exploded. Sadness is NaN or Inf.")
