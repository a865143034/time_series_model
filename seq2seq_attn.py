import json
from matplotlib import ticker
from numpy import *
from collections import Counter
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_vocab(texts, n=None):
    counter = Counter(''.join(texts))  # char level
    char2index = {w: i for i, (w, c) in enumerate(counter.most_common(n), start=4)}
    char2index['~'] = 0  # pad  不足长度的文本在后边填充0
    char2index['^'] = 1  # sos  表示句子的开头
    char2index['$'] = 2  # eos  表示句子的结尾
    char2index['#'] = 3  # unk  表示句子中出现的字典中没有的未知词
    index2char = {i: w for w, i in char2index.items()}
    return char2index, index2char

pairs = json.load(open('Time Dataset.json', 'rt', encoding='utf-8'))
print(pairs[:1])
data = array(pairs)
src_texts = data[:, 0]
trg_texts = data[:, 1]
src_c2ix, src_ix2c = build_vocab(src_texts)
trg_c2ix, trg_ix2c = build_vocab(trg_texts)


class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        # input_dim = vocab_size + 1
        self.embedding = nn.Embedding(input_dim, embedding_dim)

        self.rnn = nn.GRU(embedding_dim, hidden_dim,
                          num_layers=num_layers, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # src = [sent len, batch size]
        embedded = self.dropout(self.embedding(input_seqs))

        # embedded = [sent len, batch size, emb dim]
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)

        outputs, hidden = self.rnn(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # outputs, hidden = self.rnn(embedded, hidden)
        # outputs = [sent len, batch size, hid dim * n directions]
        # hidden = [n layers, batch size, hid dim]
        # outputs are always from the last layer
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(self.hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))
        self.v.data.normal_(mean=0, std=1. / np.sqrt(self.v.size(0)))

    def forward(self, hidden, encoder_outputs):
        #  encoder_outputs:(seq_len, batch_size, hidden_size)
        #  hidden:(num_layers * num_directions, batch_size, hidden_size)
        max_len = encoder_outputs.size(0)

        h = hidden[-1].repeat(max_len, 1, 1)
        # (seq_len, batch_size, hidden_size)

        attn_energies = self.score(h, encoder_outputs)  # compute attention score
        return F.softmax(attn_energies, dim=1)  # normalize with softmax

    def score(self, hidden, encoder_outputs):
        # (seq_len, batch_size, 2*hidden_size)-> (seq_len, batch_size, hidden_size)
        energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.permute(1, 2, 0)  # (batch_size, hidden_size, seq_len)
        v = self.v.repeat(encoder_outputs.size(1), 1).unsqueeze(1)  # (batch_size, 1, hidden_size)
        energy = torch.bmm(v, energy)  # (batch_size, 1, seq_len)
        return energy.squeeze(1)  # (batch_size, seq_len)



class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hid_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.attention = Attention(hidden_dim)
        self.rnn = nn.GRU(embedding_dim + hidden_dim, hidden_dim,
                          num_layers=num_layers, dropout=dropout)
        self.out = nn.Linear(embedding_dim + hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # input = [bsz]
        # hidden = [n layers * n directions, batch size, hid dim]
        # encoder_outputs = [sent len, batch size, hid dim * n directions]
        input = input.unsqueeze(0)
        # input = [1, bsz]
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, bsz, emb dim]
        attn_weight = self.attention(hidden, encoder_outputs)
        # (batch_size, seq_len)
        context = attn_weight.unsqueeze(1).bmm(encoder_outputs.transpose(0, 1)).transpose(0, 1)
        # (batch_size, 1, hidden_dim * n_directions)
        # (1, batch_size, hidden_dim * n_directions)
        emb_con = torch.cat((embedded, context), dim=2)
        # emb_con = [1, bsz, emb dim + hid dim]
        _, hidden = self.rnn(emb_con, hidden)
        # outputs = [sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        output = torch.cat((embedded.squeeze(0), hidden[-1], context.squeeze(0)), dim=1)
        output = F.log_softmax(self.out(output), 1)
        # outputs = [sent len, batch size, vocab_size]
        return output, hidden, attn_weight


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, teacher_forcing_ratio=0.5):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, src_seqs, src_lengths, trg_seqs):
        # src_seqs = [sent len, batch size]
        # trg_seqs = [sent len, batch size]
        batch_size = src_seqs.shape[1]
        max_len = trg_seqs.shape[0]
        trg_vocab_size = self.decoder.output_dim
        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        # hidden used as the initial hidden state of the decoder
        # encoder_outputs used to compute context
        encoder_outputs, hidden = self.encoder(src_seqs, src_lengths)
        # first input to the decoder is the <sos> tokens
        output = trg_seqs[0, :]
        for t in range(1, max_len):  # skip sos
            output, hidden, _ = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < self.teacher_forcing_ratio
            output = (trg_seqs[t] if teacher_force else output.max(1)[1])
        return outputs

    def predict(self, src_seqs, src_lengths, max_trg_len=20, start_ix=1):
        max_src_len = src_seqs.shape[0]
        batch_size = src_seqs.shape[1]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(max_trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src_seqs, src_lengths)
        output = torch.LongTensor([start_ix] * batch_size).to(self.device)
        attn_weights = torch.zeros((max_trg_len, batch_size, max_src_len))
        for t in range(1, max_trg_len):
            output, hidden, attn_weight = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            output = output.max(1)[1]
            attn_weights[t] = attn_weight
        return outputs, attn_weights