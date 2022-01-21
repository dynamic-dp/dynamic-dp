import numpy as np
import torch as t
def poisson_sampler(dataset,sampling_rate):
    batch = (np.nonzero(np.random.binomial(n = 1, p = sampling_rate ,size = len(dataset)).astype(int))[0]).tolist()
    batch_x = t.stack([t.Tensor(dataset[i][0]) for i in batch],dim = 0)
    batch_y = t.stack([t.Tensor([dataset[i][1]]) for i in batch],dim = 0)
    return batch_x,batch_y.squeeze(1).long()