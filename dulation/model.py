import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMRNN(nn.Module):
    """
    LSTMを用いた再帰型ニューラルネットワークモデル

    Args:
        in_dim (int): 入力次元数
        hidden_dim (int): 隠れ層の次元数
        output_dim (int): 出力次元数
        num_layers (int, optional): 層の数. Defaults to 1.
        bidirectional (bool, optional): 双方向にするか否か. Defaults to True.
        dropout (float, optional): ドロップアウトの割合[0.0, 1.0). Defaults to 0.0.
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 num_layers: int = 1, bidirectional: bool = True, dropout: float = 0.0):
        super().__init__()
        self.num_direction = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            in_dim,
            hidden_dim,
            num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout,
        )
        self.hidden2out = nn.Linear(self.num_direction * hidden_dim, out_dim)

    def forward(self, seqs: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        """
        順伝播.

        Args:
            seqs (torch.Tensor): 入力系列
            lens (torch.Tensor): 入力系列の長さ

        Returns:
            torch.Tensor: 出力系列
        """
        seqs = pack_padded_sequence(seqs, lens, batch_first=True)
        out, _ = self.lstm(seqs)
        out, _ = pad_packed_sequence(out, batch_first=True)
        out = self.hidden2out(out)
        return out


if __name__ == '__main__':
    import numpy as np

    data_path = '../data/dump/jsut_sr16000/org/train/in_duration/BASIC5000_0001-feats.npy'
    input_t = torch.from_numpy(np.load(data_path))  # [phoneme, feature]
    seq_len, feature_len = input_t.size()

    model = LSTMRNN(in_dim=feature_len, hidden_dim=128, out_dim=1)
    output_t = model(input_t.unsqueeze(0), [seq_len])

    print(f'入力サイズ: {input_t.shape}')
    print(f'出力サイズ: {output_t.shape}')
