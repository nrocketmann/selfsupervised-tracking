import torch
import torch.nn as nn


class MessagePassing(nn.Module):
    def __init__(self, hidden_dim, FF_hidden_dim, num_heads, key_dim):
        super().__init__()
        """
        """
        self.feedforward = FeedForward(hidden_dim, FF_hidden_dim)
        # using source_dim=target_dim=hidden_dim
        self.MHA = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

    def forward(self, sources, targets):
        """
        sources: shape (B, n, D)
        targets: shape (B, m, D)
        output shape (B, m, D)
        B = batch size
        n = num source nodes, m = num target nodes
        D = data dim (should be hidden_dim)
        """
        # note that torch.nn.MultiHeadAttention expects input order (query, key, value)
        next_targets = targets + self.MHA(targets, sources, sources, need_weights=False)[0]
        next_targets = next_targets + self.feedforward(next_targets)
        return next_targets

class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear = nn.Sequential(nn.Linear(input_dim, hidden_dim), 
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, input_dim))
    def forward(self, x):
        return self.linear(x)

class GraphMatching(nn.Module):
    def __init__(self, num_layers, hidden_dim, num_heads, FF_dim):
        super().__init__()

        self.num_layers = num_layers
        key_dim = hidden_dim // num_heads
        # require hidden_dim divisible by num_heads
        assert hidden_dim % num_heads == 0
        # require at least two hidden layer
        assert num_layers > 1

        self.mp = [MessagePassing(hidden_dim, FF_dim, num_heads, key_dim) for _ in range(num_layers)]

        #self.batchnorm = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])

    def normalize(self, hidden, layer):
        return hidden
        hidden = hidden.transpose(1, 2)
        hidden = self.batchnorm[layer](hidden)
        return hidden.transpose(1, 2)

    def forward(self, nodes1, nodes2):
        for layer in range(self.num_layers):
            next_nodes1 = self.mp[layer](nodes2, nodes1)
            nodes2 = self.mp[layer](nodes1, nodes2)
            nodes1 = next_nodes1
        
        return nodes1, nodes2


