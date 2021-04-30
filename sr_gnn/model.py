import math
import torch
from torch import nn


class GatedGNN(nn.Module):
    def __init__(self, dim_in, propagation_steps=1):
        super(GatedGNN, self).__init__()
        self._propagation_steps = propagation_steps
        self._dim_item = dim_in
        self._input_size = dim_in * 2
        self._gate_size = dim_in * 3
        self.w_ih = nn.Parameter(torch.randn(self._gate_size, self._input_size), requires_grad=True)
        self.w_hh = nn.Parameter(torch.randn(self._gate_size, self._dim_item), requires_grad=True)
        self.b_ih = nn.Parameter(torch.randn(self._gate_size), requires_grad=True)
        self.b_hh = nn.Parameter(torch.randn(self._gate_size), requires_grad=True)
        self.b_iah = nn.Parameter(torch.randn(self._dim_item), requires_grad=True)
        self.b_oah = nn.Parameter(torch.randn(self._dim_item), requires_grad=True)

        self.linear_edge_in = nn.Linear(self._dim_item, self._dim_item, bias=True)
        self.linear_edge_out = nn.Linear(self._dim_item, self._dim_item, bias=True)

        for w in [self.w_ih, self.w_hh]:
            nn.init.xavier_normal_(w, gain=nn.init.calculate_gain('relu'))

    def GNNCell(self, A, session_embedding):
        """ RNN based GNN cell to deal with the user history
        :param A: batch_size x max_n_node x (max_n_node * 2)
        :param session_embedding: batch_size x max_n_node x dim_item
        :return: batch_size x max_n_node x dim_item
        """
        # Encode the node features; batch_size x max_n_node x dim_item
        node_in_feat = self.linear_edge_in(session_embedding)
        node_out_feat = self.linear_edge_out(session_embedding)

        # Message Passing; batch_size x max_n_node x dim_item
        input_in = torch.matmul(A[:, :, :A.shape[1]], node_in_feat) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], node_out_feat) + self.b_oah

        # === GRU ===
        # batch_size x max_n_node x (dim_item * 2)
        inputs = torch.cat([input_in, input_out], dim=-1)

        # Linear Transformation: y = xW^T + b
        gi = nn.functional.linear(inputs, self.w_ih, self.b_ih)  # batch_size x max_n_node x (dim_item * 3)
        gh = nn.functional.linear(session_embedding, self.w_hh, self.b_hh)  # batch_size x max_n_node x (dim_item * 3)

        # split the tensor along with the last dim
        i_r, i_i, i_n = gi.chunk(chunks=3, dim=2)  # batch_size x max_n_node x dim_item
        h_r, h_i, h_n = gh.chunk(chunks=3, dim=2)  # batch_size x max_n_node x dim_item

        # Apply the sigmoid
        resetgate = torch.sigmoid(i_r + h_r)  # batch_size x max_n_node x dim_item
        inputgate = torch.sigmoid(i_i + h_i)  # batch_size x max_n_node x dim_item
        newgate = torch.tanh(i_n + resetgate * h_n)  # batch_size x max_n_node x dim_item
        session_embedding = newgate + inputgate * (session_embedding - newgate)  # batch_size x max_n_node x dim_item
        return session_embedding

    def forward(self, A, item_embedding):
        """
        :param A: batch_size x max_n_node x (max_n_node * 2)
        :param item_embedding: batch_size x max_n_node x dim_item
        :return: batch_size x max_n_node x dim_item
        """
        session_embedding = item_embedding
        for i in range(self._propagation_steps):
            session_embedding = self.GNNCell(A, session_embedding)  # batch_size x max_n_node x dim_item
        return session_embedding


class SessionGraph(nn.Module):
    def __init__(self, num_candidates: int, args: dict):
        super(SessionGraph, self).__init__()
        self.num_candidates = num_candidates
        self._args = args
        self._dim_item = self._args.get("dim_item", 32)
        self._hybrid = self._args.get("hybrid", False)

        # Grated GNN
        self.gated_gnn = GatedGNN(dim_in=self._dim_item, propagation_steps=self._args.get("propagation_steps", 1))

        # Embedding2Score
        self.embedding = nn.Embedding(self.num_candidates, self._dim_item)
        self.linear_one = nn.Linear(self._dim_item, self._dim_item, bias=True)
        self.linear_two = nn.Linear(self._dim_item, self._dim_item, bias=True)
        self.linear_three = nn.Linear(self._dim_item, 1, bias=False)
        self.linear_transform = nn.Linear(self._dim_item * 2, self._dim_item, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self._dim_item)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
        """
        :param hidden: batch_size x max_session_length x dim_item
        :param mask: batch_size x max_session_length
        :return:
            scores: batch_size x num_candidates
        """
        # Eq(6)
        # Get the embedding of the last item from session_embedding
        # By `torch.sum(mask, dim=1)` we get the index of the last item in a session
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, dim=1) - 1]  # batch_size x dim_item
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x dim_item
        q2 = self.linear_two(hidden)  # batch_size x max_session_length x dim_item
        alpha = self.linear_three(torch.sigmoid(q1 + q2))  # batch_size x max_session_length x 1
        s_g = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)  # batch_size x dim_item

        # Eq(7)
        if self._hybrid:
            s_h = self.linear_transform(torch.cat([s_g, ht], 1))  # TODO: is this original code correct...?
        else:
            s_h = s_g

        # Eq(8)
        b = self.embedding.weight[1:]  # num_candidates x dim_item  ; (TODO) Why do we skip the first item...?
        scores = torch.matmul(s_h, b.transpose(1, 0))  # batch_size x num_candidates
        return scores

    def forward(self, inputs):
        """
        :param inputs: tuple of the following things
            alias_inputs: batch_size x max_session_length
            A: batch_size x max_n_node x (max_n_node * 2)
            items: batch_size x max_n_node
            mask: batch_size x max_session_length
            targets: (batch_size)-sized array
        :return: scores: batch_size x num_candidates
        """
        alias_inputs, A, items, mask = inputs

        # get the item-embedding
        item_embedding = self.embedding(items)  # batch_size x max_n_node x dim_item
        session_embedding = self.gated_gnn(A, item_embedding)  # batch_size x max_n_node x dim_item

        # Embedding to Score
        # get the embeddings for items in each session; batch_size x max_session_length x dim_item
        seq_hidden = torch.stack([session_embedding[i][alias_inputs[i]] for i in range(len(alias_inputs))])
        scores = self.compute_scores(seq_hidden, mask)  # batch_size x num_candidates
        return scores


def test_GNN():
    print("test: GNN")
    batch_size = 16
    max_n_node = 10
    dim_item = 32
    device = "cpu"
    # device = "cuda"

    # batch_size x max_n_node x (max_n_node * 2)
    A = torch.randn(size=(batch_size, max_n_node, max_n_node * 2), dtype=torch.float32, device=device)
    A = (A > 0.5).type(torch.float32)
    item_embedding = torch.randn(batch_size, max_n_node, dim_item, device=device)

    gnn = GatedGNN(dim_in=dim_item).to(device=device)
    out = gnn(A, item_embedding)
    print(out.shape)


def test_SessionGraph():
    print("test: SessionGraph")
    import numpy as np

    batch_size = 16
    max_n_node = 10
    num_items = 100
    device = "cpu"
    # device = "cuda"

    # batch_size x max_n_node x (max_n_node * 2)
    A = torch.randn(size=(batch_size, max_n_node, max_n_node * 2), dtype=torch.float32, device=device)
    A = (A > 0.5).type(torch.float32)

    # Approximate the process to aggregate the items in sessions
    items = list()
    for i in range(batch_size):
        _base = [0]
        _len = np.random.randint(low=1, high=max_n_node - 1)
        _items = np.random.choice(a=num_items, size=_len, replace=False).tolist()
        _base += _items
        _base += [0] * (max_n_node - len(_base))  # 0-padding
        items.append(_base)  # batch_size x max_n_node

    items = torch.tensor(np.asarray(items), dtype=torch.long, device=device)

    model = SessionGraph(num_candidates=num_items, args=dict()).to(device=device)
    out = model(items, A)
    print(out.shape)


if __name__ == '__main__':
    test_GNN()
    test_SessionGraph()
