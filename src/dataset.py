from typing import Optional
from torch.utils.data import IterableDataset
from gluonts.dataset.common import Dataset as Dataset_gluonts
from gluonts.transform import Transformation, TransformedDataset
from gluonts.itertools import Cyclic, PseudoShuffled, Cached


class TransformedIterableDataset(IterableDataset):
    def __init__(
        self,
        dataset: Dataset_gluonts,
        transform: Transformation,
        is_train: bool = True,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = False,
    ):
        super().__init__()
        self.shuffle_buffer_length = shuffle_buffer_length

        self.transformed_dataset = TransformedDataset(
            Cyclic(dataset) if not cache_data else Cached(Cyclic(dataset)),
            transform,
            is_train=is_train,
        )

    def __iter__(self):
        if self.shuffle_buffer_length is None:
            return iter(self.transformed_dataset)
        else:
            shuffled = PseudoShuffled(
                self.transformed_dataset,
                shuffle_buffer_length=self.shuffle_buffer_length,
            )
            return iter(shuffled)


from torch.utils.data import Dataset

class TorchDataset(Dataset):
    # 构造函数
    def __init__(self, data_dic):
        self.data_dic = data_dic
    # 返回数据集大小
    def __len__(self):
        return len(self.data_dic)
    # 返回索引的数据与标签
    def __getitem__(self, index):
        return self.data_dic[index]['target']