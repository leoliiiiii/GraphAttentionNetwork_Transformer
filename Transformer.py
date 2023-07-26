# This file contains different components for a transformer model @author: Zhihan Li
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# input embedding
class Embedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)


# Module for positional Encoding
class PositionalEncoding(nn.Module):
    # d_model is equal to the dimension of the embeddings
    # max_seq_len is the size of the longest input sequence @Zhihan Li
    def __init__(self, d_model, max_seq_len, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # initialize the positional embedding matrix @Zhihan Li
        self.pe = torch.zeros(max_seq_len, d_model)
        # numerators for positional encoding @ Zhihan Li
        position = torch.arange(max_seq_len).unsqueeze(1)
        # denominators for positional encoding @Zhihan Li
        denom = torch.pow(10000, torch.arange(0, d_model, 2) / d_model)
        position_score = position / denom
        self.pe[:, 0::2] = torch.sin(position_score)
        self.pe[:, 1::2] = torch.cos(position_score)
        # additional dimension for batch @Zhihan Li
        self.pe = self.pe.unsqueeze(0)
        # add a persistent buffer to the module to save tensors that have a state but are not parameters of the model
        # buffers are not updated during backpropagation @Zhihan Li
        self.register_buffer('pe', self.pe)

    def forward(self, x):
        # increase the embedding values before addition is to make the positional encoding relatively smaller
        # in order to prevent losing the original information @ Zhihan Li
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        # indexing the positional encoding tensor to match the sequence length of the current input @Zhihan Li
        x = x + self.pe[:, :seq_len].requires_grad_(False)
        x = self.dropout(x)
        return x



# Module for multi-head attention
class MultiHeadAttention(nn.module):
    def __init__(self, d_model, heads, dropout, mask=None):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.h = heads
        assert self.d_model % self.h == 0
        self.d_k = self.d_model // self.h

        # linear transformation layers for query, key, and value @Zhihan Li
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        # batch size
        nbatch = q.size(0)

        # linear transformation on q, k, and v and split d_model dimensions into h heads,
        # where each head has (d_model // h) dimensions @Zhihan Li
        q = self.W_q(q).view(nbatch, -1, self.h, self.d_k)
        k = self.W_k(k).view(nbatch, -1, self.h, self.d_k)
        v = self.W_v(v).view(nbatch, -1, self.h, self.d_k)

        # adjust the dimension order for attention computation @Zhihan Li
        # batch size * heads * sequence length * d_k
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # compute the attention scores @Zhihan Li
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # adjust back the dimension order and concatenate the scores from all heads @Zhihan Li
        concat = scores.transpose(1, 2).contiguous().view(nbatch, -1, self.d_model)

        # go through a final linear layer @Zhihan Li
        output = self.W_o(concat)

        return output

# helper function for computing attention scores @Zhihan Li
def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    # masking
    if mask is not None:
        # the operation above generates tensors of the shape (batch, heads, n(# of queries), m(# of key-values pairs))
        # masking will mask out the last several elements from m
        scores = scores.masked_fill_(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    # dropout
    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output



# layer normalization
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a = nn.Parameter(torch.ones(d_model))
        self.b = nn.Parameter(torch.zeros(d_model))
        self.eps = eps # prevent division by zero @Zhihan Li

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # layer normalization (along the feature dimension)
        normalized = (x - mean) / (std + self.eps)
        # sometimes the network might not want the activations to be strictly normalized to zero mean and unit variance
        # self.a and self.b are learnable parameters that get adjusted during the training process via backpropagation
        # f the optimal behavior is indeed to have zero mean and unit variance,
        # the network can learn self.a close to 1 and self.b close to 0 @Zhihan Li
        output = self.a * normalized + self.b
        return output

# positionwise feed forward layer
class PositionWiseFeedFFN(nn.module):
    def __init__(self, d_model, hidden, dropout=0.1):
        self.dense1 = nn.Linear(d_model, hidden)
        self.dense2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1, ff_hidden = 1024):
        super(EncoderBlock, self).__init__()
        # multi-head attention without masking
        self.attn = MultiHeadAttention(d_model, heads, mask=None)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        # by default, the hidden layer of the position-wise feed forward is of dimension 1024
        self.ffn = PositionWiseFeedFFN(d_model, ff_hidden)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # LayerNorm (x + Sublayer(x))
        x = self.norm1(x + self.dropout1(self.attn(x, x, x)))
        x = self.norm2(x + self.dropout2(self.ffn(x)))
        return x

def clone(module, N):
    return nn.ModuleList([copy.deepcopy((module) for i in range(N))])


class Encoder(nn.module):
    def __init__(self, vocab_size, d_model, N, heads, max_seq_len):
        super(Encoder, self).__init__()
        self.N = N
        self.embed = Embedding(d_model, vocab_size)
        self.pe = PositionalEncoding(d_model, max_seq_len)
        self.encode_blocks = clone(EncoderBlock(d_model, heads), N)

    def forward(self, x):
        x = self.embed(x)
        x = self.pe(x)  # the addition operation is performed in the forward method in pe class
        for i in range(self.N):
            x = self.encode_blocks[i](x)
        return x



class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = LayerNorm(d_model)
        self.norm_2 = LayerNorm(d_model)
        self.norm_3 = LayerNorm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model)
        self.attn_2 = MultiHeadAttention(heads, d_model)
        self.ff = PositionWiseFeedFFN(d_model).cuda()


def forward(self, x, e_outputs, src_mask, trg_mask):
    x2 = self.norm_1(x)
    x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
    x2 = self.norm_2(x)
    x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs,
                                       src_mask))
    x2 = self.norm_3(x)
    x = x + self.dropout_3(self.ff(x2))
    return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads):
        super().__init__()
        self.N = N
        self.embed = Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model)
        self.layers = clone(DecoderLayer(d_model, heads), N)
        self.norm = LayerNorm(d_model)
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)


# No explicit softmax, which will automatically handled by subsequent loss function @ Zhihan Li
class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads):  # N: number of times to repeat encoder/decoder module
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads)
        self.decoder = Decoder(trg_vocab, d_model, N, heads)
        self.out = nn.Linear(d_model, trg_vocab)
    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output



