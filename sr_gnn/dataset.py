import numpy as np
import pandas as pd
from random import sample
from torch.utils.data import BatchSampler as BS, SequentialSampler as SS


def data_masks(all_usr_pois):
    us_lens = [len(upois) for upois in all_usr_pois]  # list of length of each session
    len_max = max(us_lens)  # max length of all sessions
    # 0-padding vector format of sessions with itemIds
    us_pois = [upois + [0] * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    # 0-padding vector format of sessions with 1(click) and 0(no-session)
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks


class Data(object):
    def __init__(self, data, shuffle=False, graph=None):
        inputs = data[0]  # data is the tuple of (tr_seqs, tr_labs)
        inputs, mask = data_masks(all_usr_pois=inputs)
        self.inputs = np.asarray(inputs)  # num_sessions x max_session_length
        self.mask = np.asarray(mask)  # num_sessions x max_session_length
        self.targets = np.asarray(data[1])  # (num_sessions)-size array
        self.length = len(inputs)  # num_sessions
        self.shuffle = shuffle
        self.graph = graph

    def generate_batch(self, batch_size):
        """
        :param batch_size: int
        :return: list of sessionIds for mini-batch sampling
        """
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)  # list of sessionIds for mini-batch sampling
        # deal with the remaining data in the last batch
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, i):
        """
        :param i: list of sessionIds in a mini-batch
        :return:
            alias_inputs: batch_size x max_session_length
            A: batch_size x max_n_node x (max_n_node * 2)
            items: batch_size x max_n_node
            mask: batch_size x max_session_length
            targets: (batch_size)-sized array
        """
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        items, num_candidates, A, alias_inputs = [], [], [], []
        for u_input in inputs:  # for each session in mini-batch
            # TODO: unfortunately, 0 used for 0-padding is also considered as itemId in this implementation...
            # TODO: Need to think about how to deal with 0 in `inputs` created method called `data_masks` above.
            num_candidates.append(len(np.unique(u_input)))  # get the num of unique itemIds in a session
        max_n_node = np.max(num_candidates)  # get the largest num of unique itemIds in a session of this mini-batch
        for u_input in inputs:  # for each session in mini-batch
            # TODO: unfortunately, 0 used for 0-padding is also considered as itemId in this implementation...
            # TODO: Need to think about how to deal with 0 in `inputs` created method called `data_masks` above.
            node = np.unique(u_input)  # get the unique itemIds in a session
            item = node.tolist() + (max_n_node - len(node)) * [0]  # (max_n_node)-sized array
            items.append(item)  # batch_size x max_n_node

            # Constructing the Adjacency matrix for Session Graph
            """
            Note that this adjacency matrix consists of all the items appear in the mini-batch so that
            this can be shared with other sessions and can be converted into a tensor easily when feed into a model!
            """
            u_A = np.zeros((max_n_node, max_n_node))
            for i in range(len(u_input) - 1):  # for each item in the session
                if u_input[i + 1] == 0:  # As we used 0-padding above, when it reaches 0 then we move on to next session
                    break
                u = np.where(node == u_input[i])[0][0]  # get itemId at time-step t
                v = np.where(node == u_input[i + 1])[0][0]  # get itemId at time-step t+1
                u_A[u][v] = 1

            # Normalise the Adjacency matrix for Session Graph
            # === Incoming edges
            u_sum_in = np.sum(u_A, axis=0)  # (max_n_node)-sized array
            u_sum_in[np.where(u_sum_in == 0)] = 1  # TODO: why? to avoid the division-by-0 error?
            u_A_in = np.divide(u_A, u_sum_in)

            # === Outgoing edges
            u_sum_out = np.sum(u_A, axis=1)
            u_sum_out[np.where(u_sum_out == 0)] = 1  # TODO: why? to avoid the division-by-0 error?
            u_A_out = np.divide(u_A.transpose(), u_sum_out)

            # Normalised Adjacency matrix for Session Graph for Incoming/Outgoing edges
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()  # max_n_node x (max_n_node * 2)
            A.append(u_A)  # batch_size x max_n_node x (max_n_node * 2)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])  # TODO: what is this??

        alias_inputs, A, items, mask, targets = \
            np.asarray(alias_inputs), np.asarray(A), np.asarray(items), np.asarray(mask), np.asarray(targets)
        return alias_inputs, A, items, mask, targets


class Data2(object):
    def __init__(self, df_log: pd.DataFrame, args: dict):
        self._args = args
        self._batch_size = self._args.get("batch_size", 32)
        self.df_log = df_log  # num_sessions x (max_session_length + some cols)
        self._prep()

        # for mini-batch sampling, create the list of indices and
        _sessionId_list = np.asarray(self.df_log["sessionId"].unique())
        self._cur = 0
        self._batch_ids = list()
        for i in BS(SS(sample(_sessionId_list.tolist(), k=_sessionId_list.shape[0])),
                    batch_size=self._batch_size, drop_last=False):
            self._batch_ids.append(_sessionId_list[i])

    def _prep(self):
        self.df_log = self.df_log.sort_values(by=["sessionId", "timestamp"])
        self.df_log["target"] = self.df_log["itemId"].shift(periods=1, fill_value=-1)

        # Convert the itemId into index(because itemId doesn't start with 0)
        id2index = {v: k for k, v in enumerate(self.df_log["itemId"].unique())}
        index2id = {k: v for k, v in enumerate(self.df_log["itemId"].unique())}
        self.df_log["itemId"] = self.df_log["itemId"].replace(id2index)  # TODO: this takes time.... parallelisable?

        # get the max num of sessions
        session_lens = self.df_log["sessionId"].value_counts()
        max_session_len = max(session_lens)

        self._timestep_cols = list()
        for i in range(1, max_session_len + 1):
            self._timestep_cols.append("t-{}".format(i))
            self.df_log["t-{}".format(i)] = self.df_log["itemId"].shift(periods=i, fill_value=-1)
        self._timestep_cols = sorted(self._timestep_cols)
        # print(self.df_log.head())
        # print(self.df_log.shape)

    def generate_batch(self):
        """
        :return:
        """
        sessionIds = self._batch_ids[self._cur]
        df_sample = self.df_log[self.df_log["sessionId"].isin(sessionIds)]
        num_items = len(df_sample["itemId"].unique())

        A, items, mask = list(), list(), list()
        # for each session, we construct a session-graph
        for sessionId in sorted(df_sample["sessionId"].unique()):
            # Get the corresponding rows for a session
            _df = df_sample[df_sample["sessionId"] == sessionId].sort_values(by="timestamp")
            _items = _df[self._timestep_cols].tail(1).values.tolist()[0]
            items.append(_items)

            # Binary mask indicating if there was a click at specific time-step
            _mask = [1 if i > 0 else 0 for i in _items]
            mask.append(_mask)

            # Constructing the Adjacency matrix for Session Graph
            _A = np.zeros((num_items, num_items))
            for i in range(len(_items)):
                _A[i][i + 1] = 1

            # Normalise the Adjacency matrix for Session Graph
            # === Incoming edges
            _A_sum_in = np.sum(_A, axis=0)  # (max_n_node)-sized array
            _A_sum_in[np.where(_A_sum_in == 0)] = 1  # TODO: why? to avoid the division-by-0 error?
            _A_in = np.divide(_A, _A_sum_in)

            # === Outgoing edges
            _A_sum_out = np.sum(_A, axis=1)
            _A_sum_out[np.where(_A_sum_out == 0)] = 1  # TODO: why? to avoid the division-by-0 error?
            _A_out = np.divide(_A.transpose(), _A_sum_out)

            # Normalised Adjacency matrix for Session Graph for Incoming/Outgoing edges
            _A = np.concatenate([_A_in, _A_out]).transpose()  # max_n_node x (max_n_node * 2)
            A.append(_A)  # batch_size x max_n_node x (max_n_node * 2)
        A, items, mask = np.asarray(A), np.asarray(items), np.asarray(mask)
        print(A.shape, items.shape, mask.shape)
        # alias_inputs.append([np.where(node == i)[0][0] for i in u_input])  # TODO: what is this??


def _test():
    import pickle
    print("test")

    dataset = "sample"
    batch_size = 5

    train_data = pickle.load(open('../datasets/' + dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('../datasets/' + dataset + '/test.txt', 'rb'))

    train_data = Data(data=train_data, shuffle=True)
    test_data = Data(data=test_data, shuffle=False)

    slices = train_data.generate_batch(batch_size=batch_size)
    # print(slices)

    for i in range(len(slices)):
        alias_inputs, A, items, mask, targets = train_data.get_slice(slices[i])
        print(alias_inputs.shape, A.shape, items.shape, mask.shape, targets.shape)
        print(alias_inputs)
        print(mask)
        asdf


if __name__ == '__main__':
    # _test()

    # Load the data
    df_log = pd.read_csv("../datasets/yoochoose/sample.csv")
    data = Data2(df_log=df_log, args=dict())
    data.generate_batch()
    sadf

    # 0-padding vector format of sessions with itemIds
    itemIds, masks = list(), list()
    for row in sorted(df["sessionId"].unique()):
        a = df["itemId"][df["sessionId"] == row].values
        _mask = [1 for i in a]
        _itemIds = [int(id2index[i]) for i in a]
        _itemIds += [-1] * (max_session_len - len(_itemIds))  # padding the entries for empty ids with -1
        _mask += [0] * (max_session_len - len(_mask))  # padding the entries for empty ids with 0
        assert len(_itemIds) == max_session_len
        assert len(_mask) == max_session_len
        itemIds.append(_itemIds)
        masks.append(_mask)
    itemIds, masks = np.asarray(itemIds), np.asarray(masks)
    print(itemIds.shape, masks.shape)
