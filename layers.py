"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax


class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x):
        emb = self.embed(x)   # (batch_size, seq_len, embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb

class Dropout_Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, hidden_size, drop_prob):
        super(Dropout_Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x):
        # weirdly dropout is inconsistent from q to answer

        # y = torch.tensor([[1,2,3,4,5,6,7,8,9,10]])
        # print(y)
        # yemb = self.embed(y)
        # print("yemb", yemb)
        # yemb = F.dropout(yemb, .5, self.training)
        # print("dropout y", yemb)


        emb = self.embed(x)   # (batch_size, seq_len, embed_size)
        #print("emb", emb[0])
        emb = F.dropout(emb, self.drop_prob, self.training)
        
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        #print("proj", emb[0,1:10])
        
        #print("dropped", emb[0])
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)
        emb = F.dropout2d(emb, .1, self.training)
        #print("hwy", emb[0,1:10])
        #print("hwy", emb[0])

        # dropout on embedding - how to
        # emb = F.dropout2d(emb, self.drop_prob, self.training)

        return emb

class Char_Embedding(nn.Module):
    """Char Embedding layer used by BiDAF.

    Word-level embeddings and Character-level embeddings
    are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        char_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
        char_word_size (int): number of filters for character encoding/encoding size
        window_sz (int): CNN window size
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob, XL=False, wide=True, char_word_filters_windows=[(100, 3), (150, 5), (200, 7)]):
        super(Char_Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.w_embed = nn.Embedding.from_pretrained(word_vectors)
        self.c_embed = nn.Embedding.from_pretrained(char_vectors)
        # self.convs = []
        # for filters, window_sz in char_word_filters_windows:
        #     self.convs.append(
        #         Char_CNN(char_embed_size=char_vectors.size(1), 
        #                 char_word_size=filters, 
        #                 window_sz=window_sz))
        self.wide = wide
        if wide:
            if XL:
                self.conv1 = Char_CNN(char_embed_size=char_vectors.size(1), 
                                char_word_size=300, 
                                window_sz=3)
                self.conv2 = Char_CNN(char_embed_size=char_vectors.size(1), 
                                char_word_size=400, 
                                window_sz=5)
                self.conv3 = Char_CNN(char_embed_size=char_vectors.size(1), 
                                char_word_size=200, 
                                window_sz=7)
                char_filters = 300 + 400 + 200
            else:
                self.conv1 = Char_CNN(char_embed_size=char_vectors.size(1), 
                                char_word_size=100, 
                                window_sz=3)
                self.conv2 = Char_CNN(char_embed_size=char_vectors.size(1), 
                                char_word_size=300, 
                                window_sz=5)
                self.conv3 = Char_CNN(char_embed_size=char_vectors.size(1), 
                                char_word_size=100, 
                                window_sz=7)
                char_filters = 100 + 300 + 100
        else:
            # this is for the deep character model
            self.conv_multi = Char_CNN_multi_layer(char_embed_size=char_vectors.size(1), 
                    hidden_sz=128, 
                    char_word_size=160, 
                    window_sz=5)
            char_filters = 160
        word_length = 16
        # self.max_pool = torch.nn.MaxPool1d(kernel_size=word_length - window_sz + 1)
        
        # char_filters = sum([x[0] for x in char_word_filters_windows])
        self.proj = nn.Linear(word_vectors.size(1) + char_filters, hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, w, c):
        w_emb = self.w_embed(w)   # (batch_size, seq_len, embed_size)

        c_embed = self.c_embed(c) # (batch_size, seq_len, word_length, embed_size)

        batch_size = c.size(0)
        sentence_len = c.size(1)

        c_embed = torch.transpose(c_embed, 2, 3) # (batch_size, seq_len, embed_size, word_length)
        c_embed = c_embed.view(-1, c_embed.size(2), c_embed.size(3)) # (batch_size * seq_len, embed_size, word_length)


        out = []
        # for conv in self.convs: # num_convs * (batch_size, seq_len, word_embed_size)
        #     x = conv(c_embed)
        #     x = x.view(batch_size, sentence_len, x.size(1)) # (batch_size, seq_len, word_embed_size)
        #     out.append(x)
        if self.wide:
            x = self.conv1(c_embed)
            x = x.view(batch_size, sentence_len, x.size(1))
            x = F.dropout(x, self.drop_prob, self.training)
            out.append(x)

            x = self.conv2(c_embed)
            x = x.view(batch_size, sentence_len, x.size(1))
            x = F.dropout(x, self.drop_prob, self.training)
            out.append(x)

            x = self.conv3(c_embed)
            x = x.view(batch_size, sentence_len, x.size(1))
            x = F.dropout(x, self.drop_prob, self.training)
            out.append(x)

        else:
            # this is for the deep character model
            x = self.conv_multi(c_embed)
            x = x.view(batch_size, sentence_len, x.size(1))
            x = F.dropout(x, self.drop_prob, self.training)
            out.append(x)    

        out.append(w_emb)

        emb = torch.cat(out, dim=2) # (batch_size, seq_len, word_embed_size + embed_size)

        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb

class Char_CNN(nn.Module):
    def __init__(self, char_embed_size=64, char_word_size=100, window_sz=5):
        super(Char_CNN, self).__init__()
        self.conv = nn.Conv1d(char_embed_size, char_word_size, window_sz)
        word_length = 16
        self.max_pool = nn.MaxPool1d(kernel_size=word_length - window_sz + 1)

    #   c_embed is (batch_size * seq_len, char_embed_size, word_length)
    def forward(self, c_embed):
        # c_embed = c_embed.cuda()
        conv = self.conv(c_embed)       # (batch_size * seq_len, word_embed_size, word_length - window_sz +1)
        conv = nn.functional.relu(conv)

        pool = self.max_pool(conv)  # (batch_size * seq_len, word_embed_size, 1)
        out = torch.squeeze(pool, dim=2)

        return out

class Char_CNN_multi_layer(nn.Module):
    def __init__(self, char_embed_size=64, hidden_sz=128, char_word_size=160, window_sz=5):
        super(Char_CNN_multi_layer, self).__init__()
        self.conv = nn.Conv1d(char_embed_size, hidden_sz, window_sz)
        self.conv2 = nn.Conv1d(hidden_sz, char_word_size, window_sz)
        word_length = 16
        self.max_pool = nn.MaxPool1d(kernel_size=word_length - window_sz + 1 - window_sz + 1)

    #   c_embed is (batch_size * seq_len, char_embed_size, word_length)
    def forward(self, c_embed):
        # c_embed = c_embed.cuda()
        conv = self.conv(c_embed)       # (batch_size * seq_len, word_embed_size, word_length - window_sz +1)
        conv = nn.functional.relu(conv)

        conv2 = self.conv2(conv) 
        conv2 = nn.functional.relu(conv2)

        pool = self.max_pool(conv2)  # (batch_size * seq_len, word_embed_size, 1)
        out = torch.squeeze(pool, dim=2)

        return out

class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x

class PointNet(nn.Module):
    def __init__(self,
                hidden_size,
                kernel_size=1):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(hidden_size,200, kernel_size = 1)
        # self.conv2 = nn.Conv1d(200, 400, kernel_size = 1)
        self.conv3 = nn.Conv1d(200, 400, kernel_size = 1)

        self.bn1 = nn.BatchNorm1d(200)
        #self.bn2 = nn.BatchNorm1d(400)
        self.bn3 = nn.BatchNorm1d(400)

        # self.proj1 = nn.Linear(900, 200)
        # self.proj2 = nn.Linear(500, 200)

    def forward(self, emb, enc_h):
        # emb -- tensor of shape (batch_size, seq_len, hidden_size)

        seq_len = emb.size()[1]
        emb = emb.permute(0,2,1)
        emb = F.relu(self.bn1(self.conv1(emb)))

        # emb = F.relu(self.bn2(self.conv2(emb)))
        emb = self.bn3(self.conv3(emb))
        point_feats = torch.max(emb, 2, keepdim=True)[0].squeeze(-1)

        # attatch hidden layer, project 
        # global_feats = torch.cat((point_feats, enc_h), 1)
        # global_feats_proj1 = F.relu(self.proj1(global_feats))
        # global_feats_proj1 = F.dropout(global_feats_proj1, .2, self.training)
        # global_feats_proj2 = F.relu(self.proj2(global_feats_proj1))
        # global_feats_proj2 = F.dropout(global_feats_proj2, .2, self.training)

        return point_feats

class WordCNN(nn.Module):
    def __init__(self,
                hidden_size,
                kernel_size=5,
                padding=2):
        super(WordCNN, self).__init__()
        self.conv1 = nn.Conv1d(hidden_size, 200, kernel_size = 5, padding=2)
        #self.conv2 = nn.Conv1d(200, 400, kernel_size = 5, padding=2)
        self.conv3 = nn.Conv1d(200, 400, kernel_size = 3, padding=1)

        self.bn1 = nn.BatchNorm1d(200)
        #self.bn2 = nn.BatchNorm1d(400)
        self.bn3 = nn.BatchNorm1d(400)

        # self.proj1 = nn.Linear(800, 200)
        # self.proj2 = nn.Linear(500, 200)

    def forward(self, emb):
        # emb -- tensor of shape (batch_size, seq_len, hidden_size)

        seq_len = emb.size()[1]
        emb = emb.permute(0,2,1)
        emb = F.relu(self.bn1(self.conv1(emb)))

        #emb = F.relu(self.bn2(self.conv2(emb)))
        emb = self.bn3(self.conv3(emb))       
        emb = emb.permute(0,2,1)
        # attatch hidden layer, project 
        # global_feats_proj1 = F.relu(self.proj1(emb))
        # global_feats_proj1 = F.dropout(global_feats_proj1, .2, self.training)
        # global_feats_proj2 = F.relu(self.proj2(global_feats_proj1))
        # global_feats_proj2 = F.dropout(global_feats_proj2, .2, self.training)

        return emb


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths, batch_first=True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)

        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)


        return x


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s

class SelfAttention(nn.Module):
    """Self-attention accoridng to R-Net.

    Computes self-attention for context words across entire context?


    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(SelfAttention, self).__init__()
        self.drop_prob = drop_prob
        self.W_self = nn.Linear(hidden_size, hidden_size)
        self.W_other = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Parameter(torch.zeros(hidden_size))
        for weight in (self.v):
            nn.init.normal_(weight)


    def forward(self, c, c_mask):
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size) the word in question
        batch_size, c_len, _ = c.size()

        proj_selves = self.W_self(c) # (batch_sz, c_len, hidden_sz)
        proj_others = self.W_other(c) # (batch_sz, c_len, hidden_sz)

        proj_selves = proj_selves.unsqueeze(2) # (batch_sz, c_len, 1,  hidden_sz)
        proj_selves = proj_selves.expand(-1, -1, c_len, -1) # (batch_sz, c_len, c_len, hidden_sz)

        proj_others = proj_others.unsqueeze(1) # (batch_sz, 1, c_len, hidden_sz)

        combined = proj_selves + proj_others # (batch_sz, c_len, c_len, hidden_sz) 
        # (context words intexed by dim 1, other words by dim 2)

        combined = torch.tanh(combined) # (batch_sz, c_len, c_len, hidden_sz) 

        s = torch.matmul(combined, self.v) # (batch_sz, c_len, c_len)
        c_mask = c_mask.view(batch_size, 1, -1) 

        a = masked_softmax(s, c_mask, dim=2) # (batch_sz, c_len, c_len)

        c = torch.bmm(a, c) # (batch_sz, c_len, hidden_sz)
        return c


class GlobalBiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(GlobalBiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.global_proj = nn.Linear(400,200)

        # self.cg_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        # self.qg_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        # self.cqg_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))

        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask, q_global, c_conv):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)


        # make q_global the proper size (batch_size, 900) --> (batch_size, 1, 900)
        q_global = q_global.unsqueeze(1)        # (batch_size, 1, 900)
        # elementwise product of q_clobal and c_conv
        global_sim = q_global*c_conv            # (batch_size, c_len, 900)
        # project to size 200
        global_sim = self.global_proj(global_sim)       # (batch_size, c_len, 200)
        # add to vector x
        x = torch.cat([c, a, c * a, c * b, global_sim], dim=2)  # (bs, c_len, 5 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))

        # g = g.unsqueeze(1)

        # s3 = torch.matmul(c * g, self.cg_weight).expand([-1, -1, q_len])
        # s4 = torch.matmul(q * g, self.qg_weight).transpose(1, 2)\
        #                                    .expand([-1, c_len, -1])
        # s5 = torch.matmul(c * g * self.cqg_weight, q.transpose(1, 2))

        s = s0 + s1 + s2 + self.bias

        return s


class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob, att_size=None):
        super(BiDAFOutput, self).__init__()
        if att_size == None:
            att_size=8*hidden_size

        self.att_linear_1 = nn.Linear(10*hidden_size, 1) # i changed this

        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(10*hidden_size, 1) # i changed this
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        # print("att", att.shape)
        # print("mod", mod.shape)
        # test1= self.att_linear_1(att)
        # test2=self.att_linear_2(att)
        # print(test1.shape)
        # print(test2.shape)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2
