import random
from pathlib import Path

from hydra.utils import to_absolute_path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def fix_seed(seed: int) -> None:
    """
    再現性の担保のために乱数シードの固定する関数.

    Args:
        seed (int): シード値
    """
    random.seed(seed)     # random
    np.random.seed(seed)  # numpy
    # pytorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_utt_list(utt_list: str) -> list:
    """
    Load a list of utterances.

    Args:
        utt_list (str): path to a file containing a list of utterances

    Returns:
        List[str]: list of utterances
    """
    utt_ids = []
    with open(utt_list) as f:
        for utt_id in f:
            utt_id = utt_id.strip()
            if len(utt_id) > 0:
                utt_ids.append(utt_id)
    return utt_ids


def pad_2d(x: torch.Tensor, max_len: int, constant_values: int = 0):
    """
    Pad a 2d-tensor.

    Args:
        x (torch.Tensor): tensor to pad
        max_len (int): maximum length of the tensor
        constant_values (int, optional): value to pad with. Default: 0

    Returns:
        torch.Tensor: padded tensor
    """
    # 短い系列は後ろにパディングして最長系列の長さに揃える
    x = np.pad(
        x,
        [(0, max_len - len(x)), (0, 0)],
        mode="constant",
        constant_values=constant_values,
    )
    return x


def collate_fn_dulation(batch):
    """ 短い系列は最長系列の長さに揃えた上で,各データをtensorにして返す """
    lengths = [len(x[0]) for x in batch]
    lengths = torch.tensor(lengths, dtype=torch.long)
    max_len = max(lengths)

    x_batch = torch.stack([torch.from_numpy(pad_2d(x[0], max_len)) for x in batch])
    y_batch = torch.stack([torch.from_numpy(pad_2d(x[1], max_len)) for x in batch])

    return x_batch, y_batch, lengths


def get_dataloaders(data_config):
    data_loaders = {}

    for phase in ['train', 'dev']:
        utt_ids = load_utt_list(to_absolute_path(data_config[phase].utt_list))
        in_dir = Path(to_absolute_path(data_config[phase].in_dir))
        out_dir = Path(to_absolute_path(data_config[phase].out_dir))

        in_feats_paths = [in_dir / f'{utt_id}-feats.npy' for utt_id in utt_ids]
        out_feats_paths = [out_dir / f'{utt_id}-feats.npy' for utt_id in utt_ids]

        dataset = DulationDataset(in_feats_paths, out_feats_paths)
        data_loaders[phase] = DataLoader(
            dataset,
            batch_size=data_config.batch_size,
            shuffle=phase.startswith("train"),
            num_workers=data_config.num_workers,
            collate_fn=collate_fn_dulation,
            pin_memory=True,
        )

    return data_loaders


class DulationDataset(Dataset):
    def __init__(self, in_paths, out_paths):
        self.in_paths = in_paths
        self.out_paths = out_paths

    def __getitem__(self, idx):
        data = np.load(self.in_paths[idx])
        label = np.load(self.out_paths[idx])
        return data, label

    def __len__(self):
        return len(self.in_paths)


if __name__ == '__main__':
    data_dir = Path('../data/dump/jsut_sr16000/org/train/')
    in_paths = list(data_dir.glob('in_dulation/*.npy'))
    out_paths = list(data_dir.glob('out_dulation/*.npy'))

    dataset = DulationDataset(in_paths, out_paths)
    dataloader = DataLoader(dataset, batch_size=8,
                            collate_fn=collate_fn_dulation, num_workers=0)
    print(f'データ数: {len(dataset)}')
    print(f'イテレーション数: {len(dataloader)}')

    in_feats, out_feats, lengths = next(iter(dataloader))
    print(f'入力特徴量のサイズ: {in_feats.shape}')  # [batch, phoneme, feature]
    print(f'出力特徴量のサイズ: {out_feats.shape}')
    print(f'系列長のサイズ: {lengths.shape}')
