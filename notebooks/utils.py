
import numpy as np

def batch_generator(batch_size, data, labels=None):
    """generate batches of samples"""
    n_batches = int(np.ceil(len(data) / float(batch_size)))
    idx = np.random.permutation(len(data))
    data_shuffled = data[idx]
    if labels is not None:
        labels_shuffled = labels[idx]
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        if labels is not None:
            yield data_shuffled[start:end, :], labels_shuffled[start:end]
        else:
            yield data_shuffled[start:end, :]

def gen_batches(data, batch_size):
    data = np.array(data)
    for i in range(0, data.shape[0], batch_size):
        yield data[i: i + batch_size]
