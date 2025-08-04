Dataset Wrapping  
MindSpore provides parsing and reading for various typical open-source datasets, such as MNIST, CIFAR-10, CLUE, LJSpeech, etc. For details, refer to mindspore.dataset.

Custom Data Loading with GeneratorDataset  
In migration scenarios, the most commonly used data loading method is GeneratorDataset. You only need to simply wrap a Python iterator, and you can directly connect it to MindSpore models for training and inference.
```python
import numpy as np
from mindspore import dataset as ds

num_parallel_workers = 2  # Number of threads/processes
world_size = 1            # Used in parallel scenarios, communication group_size
rank = 0                  # Used in parallel scenarios, communication rank_id

class MyDataset:
    def __init__(self):
        self.data = np.random.sample((5, 2))
        self.label = np.random.sample((5, 1))

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

dataset = ds.GeneratorDataset(source=MyDataset(), column_names=["data", "label"],
                              num_parallel_workers=num_parallel_workers, shuffle=True,
                              num_shards=1, shard_id=0)
train_dataset = dataset.batch(batch_size=2, drop_remainder=True, num_parallel_workers=num_parallel_workers)

```

A typical dataset construction is as above: create a Python class that must have __getitem__ and __len__ methods, representing the data fetched at each iteration and the total size of the dataset, respectively. The index indicates the data index fetched each time; when shuffle=False, it increases sequentially, and when shuffle=True, it is randomly shuffled.

GeneratorDataset must include at least:

source: a Python iterator;

column_names: the names of each output from the iterator's __getitem__ method.

dataset.batch combines consecutive batch_size data entries in the dataset into a batch. It must include at least:

batch_size: specifies the number of data entries in each batch.

Main differences between MindSpore's GeneratorDataset and PyTorch's DataLoader:

- MindSpore's GeneratorDataset must be passed column_names;
- PyTorch's data augmentation operates on Tensor objects, while MindSpore's operates on numpy objects, and data processing cannot use MindSpore's mint, ops, or nn operators;
- PyTorch's batch operation is an attribute of DataLoader, while MindSpore's batch operation is a separate