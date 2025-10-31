import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, n_layers, cell_type='GRU', dropout=0.1):
        super(Encoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        
        rnn_cell = getattr(nn, cell_type)
        
        self.rnn = rnn_cell(emb_dim, hidden_dim, n_layers, 
                            dropout=dropout, batch_first=True, 
                            bidirectional=True)
        
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)
        self.cell_type = cell_type

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        
        encoder_outputs, hidden = self.rnn(embedded)
        
        if self.cell_type == 'LSTM':
            h_n, c_n = hidden
            h_n = h_n.view(self.n_layers, 2, -1, self.hidden_dim).permute(0, 2, 1, 3).contiguous()
            c_n = c_n.view(self.n_layers, 2, -1, self.hidden_dim).permute(0, 2, 1, 3).contiguous()
            
            h_n = self.fc_hidden(h_n.view(self.n_layers, -1, self.hidden_dim * 2))
            c_n = self.fc_cell(c_n.view(self.n_layers, -1, self.hidden_dim * 2))
            hidden = (h_n, c_n)
        else:
            hidden = hidden.view(self.n_layers, 2, -1, self.hidden_dim).permute(0, 2, 1, 3).contiguous()
            hidden = self.fc_hidden(hidden.view(self.n_layers, -1, self.hidden_dim * 2))
            
        return encoder_outputs, hidden

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim

    def forward(self, decoder_hidden, encoder_outputs):
        attn_scores = torch.sum(encoder_outputs, dim=2)
        return F.softmax(attn_scores, dim=1)


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, n_layers, cell_type='GRU', dropout=0.1):
        super(Decoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        
        rnn_cell = getattr(nn, cell_type)
        
        self.rnn = rnn_cell(emb_dim, hidden_dim, n_layers, 
                            dropout=dropout, batch_first=True)
        
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden_state):
        embedded = self.embedding(input_token)
        embedded = self.dropout(embedded)
        
        output, hidden = self.rnn(embedded, hidden_state)
        
        prediction = self.fc_out(output.squeeze(1))
        
        return F.log_softmax(prediction, dim=1), hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.vocab_size
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(src.device)
        
        encoder_outputs, hidden = self.encoder(src)
        
        input_token = trg[:, 0].unsqueeze(1)
        
        for t in range(1, trg_len):
            output, hidden = self.decoder(input_token, hidden)
            
            outputs[:, t] = output
            
            if torch.rand(1) < teacher_forcing_ratio:
                input_token = trg[:, t].unsqueeze(1)
            else:
                input_token = output.argmax(1).unsqueeze(1)
                
        return outputs
