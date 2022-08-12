from pathlib import Path

import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

from dataloader import get_dataloaders, fix_seed
from model import LSTMRNN


def setup(config: dict, device: torch.device) -> tuple:
    """
    学習のためのセットアップをする関数.

    Args:
        config (dict): 学習のためのコンフィグ
        device (torch.device): 使用するデバイス

    Returns:
        tuple: dataloader, model, criterion, optimizer, lr_scheduler, and writer
    """
    # シード値の固定
    fix_seed(config.seed)

    # CUDAの設定
    if torch.cuda.is_available():
        from torch.backends import cudnn
        cudnn.benchmark = config.cudnn.benchmark
        cudnn.deterministic = config.cudnn.deterministic

    # データローダとモデルと損失関数
    dataloaders = get_dataloaders(config.data)
    model = LSTMRNN(**config.model.netG).to(device)
    criterion = nn.MSELoss()

    # 最適化手法とスケジューラ(ライブラリからクラス名でヒットさせて取得)
    optimizer_class = getattr(optim, config.train.optim.optimizer.name)
    optimizer = optimizer_class(
        model.parameters(), **config.train.optim.optimizer.params
    )
    lr_scheduler_class = getattr(optim.lr_scheduler, config.train.optim.lr_scheduler.name)
    lr_scheduler = lr_scheduler_class(optimizer, gamma=0.5, step_size=10)

    # Tensorboard
    writer = SummaryWriter(to_absolute_path(config.train.log_dir))

    # configファイルを保存しておく
    out_dir = Path(to_absolute_path(config.train.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'model.yaml', 'w') as f:
        OmegaConf.save(config.model, f)
    with open(out_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(config, f)

    return dataloaders, model, criterion, optimizer, lr_scheduler, writer


@hydra.main(config_path='conf', config_name='config')
def main(config: DictConfig) -> None:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataloaders, model, criterion, optimizer, lr_scheduler, writer = setup(config, device)


if __name__ == '__main__':
    main()
    # # train
    # for in_feats, out_feats, lengths in dataloader:
    #     pred_out_feats = model(in_feats, lengths)
    #     loss = criterion(pred_out_feats, out_feats)

    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
