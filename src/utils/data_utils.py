import random
import torch.utils.data as data


class BatchSampler(data.Sampler):
    def __init__(self, dataset, node_num=None, max_len=5000, batch_token_num=3096, shuffle=True):
        if node_num is None:
            self.node_num = dataset.get_lenth()
        else:
            self.node_num = node_num
        self.idx = [i for i in range(len(self.node_num))  
                        if self.node_num[i] <= max_len]
        self.shuffle = shuffle
        self.batches = []
        self.max_len = max_len
        self.batch_token_num = batch_token_num
    
    def _form_batches(self):
        if self.shuffle: random.shuffle(self.idx)
        idx = self.idx
        while idx:
            batch = []
            n_nodes = 0
            while idx and n_nodes + self.node_num[idx[0]] <= self.batch_token_num:
                next_idx, idx = idx[0], idx[1:]
                n_nodes += self.node_num[next_idx]
                batch.append(next_idx)
            self.batches.append(batch)
    
    def __len__(self): 
        if not self.batches: self._form_batches()
        return len(self.batches)
    
    def __iter__(self):
        if not self.batches: self._form_batches()
        for batch in self.batches: yield batch