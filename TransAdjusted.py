import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, x, timestamps, device):  # timestamps refer to the pos from previous steps in GATE
        pe = torch.zeros(len(timestamps), self.d_model)
        # Expand dimensions
        timestamps = timestamps.float().unsqueeze(-1)

        # Create a range of dimension indices
        dim_indices = torch.arange(self.d_model, dtype=torch.float)

        # Create a denominator for the positional encoding calculation
        denom = torch.pow(10000, (2 * dim_indices) / self.d_model).to(device)

        # Apply sine to even indices in the timestamp / denom
        pe[:, 0::2] = torch.sin(timestamps / denom[0::2])

        # Apply cosine to odd indices in the timestamp / denom
        pe[:, 1::2] = torch.cos(timestamps / denom[1::2])

        # Add the positional encoding to the token embeddings
        x = x + pe.to(device)
        x = self.dropout(x)

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads, dropout, mask_future_positions=False):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.h = heads
        assert self.d_model % self.h == 0
        self.d_k = self.d_model // self.h
        self.mask_future_positions = mask_future_positions

        # linear transformation layers for query, key, and value @Zhihan Li
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, device, mask=None):
        # linear transformation on q, k, and v and split d_model dimensions into h heads,
        # where each head has (d_model // h) dimensions @Zhihan Li
        q = self.W_q(q).view(-1, self.h, self.d_k)
        k = self.W_k(k).view(-1, self.h, self.d_k)
        v = self.W_v(v).view(-1, self.h, self.d_k)

        # adjust the dimension order for attention computation @Zhihan Li
        # batch size * heads * sequence length * d_k
        k = k.transpose(0, 1)
        q = q.transpose(0, 1)
        v = v.transpose(0, 1)

        # compute the attention scores @Zhihan Li
        scores = attention(q, k, v, self.d_k, device, mask, self.dropout, self.mask_future_positions)

        # adjust back the dimension order and concatenate the scores from all heads @Zhihan Li
        concat = scores.transpose(0, 1).contiguous().view(-1, self.d_model)

        # go through a final linear layer @Zhihan Li
        output = self.W_o(concat)

        return output

# helper function for computing attention scores @Zhihan Li
def attention(q, k, v, d_k, device, mask=None, dropout=None, mask_future_positions=False):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    # Masking out future positions for masked MultiHeadAttention Module in decoder
    if mask_future_positions:
        future_position_mask = torch.triu(torch.ones((scores.size(-2), scores.size(-1))), diagonal=1).bool().to(device)
        scores.masked_fill_(future_position_mask, float('-inf'))

    # for masking out paddings
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
class PositionWiseFeedFFN(nn.Module):
    def __init__(self, d_model, hidden, dropout=0.1):
        super(PositionWiseFeedFFN, self).__init__()
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
        self.attn = MultiHeadAttention(d_model, heads, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        # by default, the hidden layer of the position-wise feed forward is of dimension 1024
        self.ffn = PositionWiseFeedFFN(d_model, ff_hidden)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, device):
        # LayerNorm (x + Sublayer(x))
        x = self.norm1(x + self.dropout1(self.attn(x, x, x, device)))
        x = self.norm2(x + self.dropout2(self.ffn(x)))
        return x

def clone(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, d_model, N, heads, trans_dropout):
        super(Encoder, self).__init__()
        self.N = N
        # self.embed = Embedding(d_model, vocab_size) # embedding module already exists before GAM
        self.pe = PositionalEncoding(d_model, trans_dropout)
        self.encoder_blocks = clone(EncoderBlock(d_model, heads, trans_dropout), N)

    def forward(self, x, pos, device):
        x = self.pe(x, pos, device)  # the addition operation is performed in the forward method in pe class
        for i in range(self.N):
            x = self.encoder_blocks[i](x, device)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1, ff_hidden = 1024):
        super(DecoderBlock, self).__init__()
        # masked MultiHeadAttention block
        self.masked_attn = MultiHeadAttention(d_model, heads, dropout, mask_future_positions=True)
        # unmasked MultiHeadAttention block
        self.attn = MultiHeadAttention(d_model, heads, dropout)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        self.ffn = PositionWiseFeedFFN(d_model, ff_hidden)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_out, device):
        # masked attn layer
        x = self.norm1(x + self.dropout1(self.masked_attn(x, x, x, device)))
        # attn layer (taking both encoder output (K, V) and data (Q) input as inputs
        x = self.norm2(x + self.dropout2(self.attn(x, enc_out, enc_out, device)))
        # position-wise feed forward
        x = self.norm3(x + self.dropout3(self.ffn(x)))
        return x

class Decoder(nn.Module):
    def __init__(self, d_model, N, heads, trans_dropout):
        super(Decoder, self).__init__()
        self.N = N
        self.pe = PositionalEncoding(d_model, trans_dropout)
        self.decoder_blocks = clone(DecoderBlock(d_model, heads, trans_dropout), N)
        self.norm = LayerNorm(d_model)
    def forward(self, x, enc_out, pos, device):
        x = self.pe(x, pos, device)  # the addition operation is performed in the forward method in pe class
        for i in range(self.N):
            x = self.decoder_blocks[i](x, enc_out, device)
        return x


# No explicit softmax, which will automatically handled by subsequent loss function @ Zhihan Li
class Transformer(nn.Module):
    def __init__(self, d_model, N, heads, out_dim=10, trans_dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(d_model, N, heads, trans_dropout)
        self.decoder = Decoder(d_model, N, heads, trans_dropout)
        self.linear = nn.Linear(d_model, out_dim)
        # self.sigmoid = nn.Sigmoid()
    def forward(self, x, pos, device):
        enc_out = self.encoder(x, pos, device)
        dec_out = self.decoder(x, enc_out, pos, device)
        out = self.linear(dec_out)
        return out




