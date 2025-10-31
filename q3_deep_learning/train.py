import torch
import torch.nn as nn
import torch.optim as optim
from model import Encoder, Decoder, Seq2Seq
import numpy as np

INPUT_DIM = 100
OUTPUT_DIM = 120
EMB_DIM = 128
HID_DIM = 256
N_LAYERS = 1
CELL_TYPE = 'GRU'

enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, CELL_TYPE)
dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, CELL_TYPE)
model = Seq2Seq(enc, dec)

optimizer = optim.Adam(model.parameters())
criterion = nn.NLLLoss(ignore_index=0)

def train_loop(model, optimizer, criterion):
    model.train()
    epoch_loss = 0
    
    src = torch.randint(1, INPUT_DIM, (32, 10))
    trg = torch.randint(1, OUTPUT_DIM, (32, 10))

    optimizer.zero_grad()
    
    output = model(src, trg)
    
    output_dim = output.shape[-1]
    
    output = output[:, 1:].reshape(-1, output_dim)
    trg = trg[:, 1:].reshape(-1)
    
    loss = criterion(output, trg)
    
    loss.backward()
    
    optimizer.step()
    
    return loss.item()

if __name__ == "__main__":
    print("Starting training loop...")
    loss = train_loop(model, optimizer, criterion)
    print(f"Loss after one batch: {loss}")
    
    if np.isnan(loss) or np.isinf(loss):
        print("\nTraining failed: Loss is NaN or Inf.")
