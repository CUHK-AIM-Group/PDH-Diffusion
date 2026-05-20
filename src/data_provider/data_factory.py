from data_provider.data_loader_hcp import Dataset_fmri
from torch.utils.data import DataLoader

data_dict = {
    'hcp': Dataset_fmri,
}

# def data_provider(args, flag):
def data_provider(args, flag, data):
    Data = data_dict['hcp']
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'val':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size  # bsz=1 for evaluation

    elif flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size  # bsz=1 for evaluation

    else:
        # shuffle_flag = True
        shuffle_flag = False #为了确保随机性改为了false 20241209
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid

    data_set = Data(
        root_path=args.root_path,
        data=data,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        drop_short=drop_last,  # drop too short sequences in dataset
    )
    # print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
