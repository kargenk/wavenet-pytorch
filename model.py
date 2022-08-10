from packaging import version

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dsp import mulaw_quantize


def conv1d(in_channels, out_channels, kernel_size, *args, **kwargs):
    """Weight-normalized Conv1d layer."""
    out = Conv1d(in_channels, out_channels, kernel_size, *args, **kwargs)
    return nn.utils.weight_norm(out)


def conv1d1x1(in_channels, out_channels, bias=True):
    return conv1d(in_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=bias)


class Conv1d(nn.Conv1d):
    """
    Extended nn.Conv1d for incremental dilated convolutions.
    ref: https://github.com/r9y9/ttslearn/blob/master/ttslearn/wavenet/conv.py
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clear_buffer()
        self._linearized_weight = None

        torch_is_ge_180 = version.parse(torch.__version__) >= version.parse("1.8.0")
        if torch_is_ge_180:
            self.register_full_backward_hook(self._clear_linearized_weight)
        else:
            self.register_backward_hook(self._clear_linearized_weight)

    def incremental_forward(self, input):
        # input: (B, T, C)
        if self.training:
            raise RuntimeError("incremental_forward only supports eval mode")

        # run forward pre hooks (e.g., weight norm)
        for hook in self._forward_pre_hooks.values():
            hook(self, input)

        # reshape weight
        weight = self._get_linearized_weight()
        kw = self.kernel_size[0]
        dilation = self.dilation[0]

        bsz = input.size(0)  # input: bsz x len x dim
        if kw > 1:
            input = input.data
            if self.input_buffer is None:
                self.input_buffer = input.new(
                    bsz, kw + (kw - 1) * (dilation - 1), input.size(2)
                )
                self.input_buffer.zero_()
            else:
                # shift buffer
                self.input_buffer[:, :-1, :] = self.input_buffer[:, 1:, :].clone()
            # append next input
            self.input_buffer[:, -1, :] = input[:, -1, :]
            input = self.input_buffer
            if dilation > 1:
                input = input[:, 0::dilation, :].contiguous()
        with torch.no_grad():
            output = F.linear(input.view(bsz, -1), weight, self.bias)
        return output.view(bsz, 1, -1)

    def clear_buffer(self):
        self.input_buffer = None

    def _get_linearized_weight(self):
        if self._linearized_weight is None:
            kw = self.kernel_size[0]
            # nn.Conv1d
            assert self.weight.size() == (self.out_channels, self.in_channels, kw)
            weight = self.weight.transpose(1, 2).contiguous()
            assert weight.size() == (self.out_channels, kw, self.in_channels)
            self._linearized_weight = weight.view(self.out_channels, -1)
        return self._linearized_weight

    def _clear_linearized_weight(self, *args):
        self._linearized_weight = None


class UpsampleNet(nn.Module):
    def __init__(self, upsample_scales):
        super().__init__()
        self.upsample_scales = upsample_scales
        self.conv_layers = nn.ModuleList()
        for scale in upsample_scales:
            kernel_size = (1, scale * 2 + 1)
            conv = nn.Conv2d(1, 1, kernel_size, padding=(0, scale), bias=False)  # melspなどの2次元特徴量も考慮して2D
            # 畳み込み前後で入力と出力のスケール保持のため，重みをカーネルサイズの逆数で初期化
            conv.weight.data.fill_(1.0 / np.prod(kernel_size))
            self.conv_layers.append(conv)

    def forward(self, condition):
        c = condition.unsqueeze(1)  # [N, C, T] -> [N, 1, C, T]

        # 最近傍補間と畳み込みの繰り返し
        for idx, scale in enumerate(self.upsample_scales):
            # 時間方向のみアップサンプリング([N, 1, C, T] -> [N, 1, C, T*scale])
            c = F.interpolate(c, scale_factor=(1, scale), mode='nearest')
            c = self.conv_layers[idx](c)

        return c.squeeze(1)


class UpsampleConvNet(nn.Module):
    def __init__(self, upsample_scales, cin_channels, aux_context_window):
        super().__init__()

        # 1Dconvを用いて条件付け特徴量の近傍を考慮
        kernel_size = 2 * aux_context_window + 1
        self.conv_in = conv1d(cin_channels, cin_channels, kernel_size, bias=False)
        self.upsample = UpsampleNet(upsample_scales)

    def forward(self, c):
        return self.upsample(self.conv_in(c))


class DilatedResSkipBlock(nn.Module):
    def __init__(self, res_channels, gate_channels, skip_channels, kernel_size,
                 dilation=1, cin_channels=80, *args, **kwargs,):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation

        # 1D dilated conv
        self.conv = conv1d(res_channels, gate_channels, kernel_size,
                              padding=self.padding, dilation=dilation, *args, **kwargs)

        # local condition用の次元調整
        self.local_cond = conv1d1x1(cin_channels, gate_channels, bias=False)

        # ゲート周りの次元調整
        gate_out_channels = gate_channels // 2
        self.skip_out = conv1d1x1(gate_out_channels, skip_channels)
        self.out = conv1d1x1(gate_out_channels, res_channels)

    def forward(self, x, condition):
        residual = x

        # 因果的(causal)畳み込み
        x = self.conv(x)
        x = x[:, :, : -self.padding]           # 因果性を保障するために出力をシフト
        a, b = x.split(x.size(1) // 2, dim=1)  # チャネル方向をゲート用に2分割

        # local conditioning(時間変化する特徴量: 言語特徴で条件付け)
        c = self.local_cond(condition)
        ca, cb = c.split(c.size(1) // 2, dim=1)  # チャネル方向をゲート用に2分割
        a, b = a + ca, b + cb

        # ゲート機構
        x = torch.tanh(a) * torch.sigmoid(b)

        # スキップ接続
        s = self.skip_out(x)

        # 残差(residual)接続
        x = self.out(x)
        x = x + residual

        return x, s


class WaveNet(nn.Module):
    def __init__(self, out_channels=256, layers=30, stacks=3, res_channels=64,
                 gate_channels=128, skip_channels=64, kernel_size=2,
                 cin_channels=80, upsample_scales=None, aux_context_window=0):
        super().__init__()
        self.out_channels = out_channels

        if upsample_scales is None:
            upsample_scales = [10, 8]

        self.first_conv = conv1d1x1(out_channels, res_channels)

        self.main_conv_layers = nn.ModuleList()
        layers_per_stack = layers // stacks
        for layer in range(layers):
            dilation = 2 ** (layer % layers_per_stack)
            conv = DilatedResSkipBlock(res_channels, gate_channels, skip_channels, kernel_size,
                                       dilation=dilation, cin_channels=cin_channels)
            self.main_conv_layers.append(conv)

        # スキップ接続の総和から波形への変換
        self.last_conv_layers = nn.ModuleList(
            [
                nn.ReLU(),
                conv1d1x1(skip_channels, skip_channels),
                nn.ReLU(),
                conv1d1x1(skip_channels, out_channels),
            ]
        )

        # フレーム単位の特徴量をサンプル単位にアップサンプリング
        self.upsample_net = UpsampleConvNet(upsample_scales, cin_channels, aux_context_window)

    def forward(self, x, condition):
        # 量子化された離散数列からOne-hotベクトルに変換
        # [N, T] -> [N, T, out_channels] -> [N, out_channels, T]
        x = F.one_hot(x, self.out_channels).transpose(1, 2).float()
        x = self.first_conv(x)

        # 条件付け特徴量のアップサンプリング
        c = self.upsample_net(condition)

        # メインのDilated畳み込み
        skips = 0
        for f in self.main_conv_layers:
            x, h = f(x, c)
            skips += h  # 各層におけるスキップ接続の出力を加算して保持

        # スキップ接続の和を入力として出力を計算
        x = skips
        for f in self.last_conv_layers:
            x = f(x)

        return x

    def inference(self, condition: torch.Tensor, num_time_steps: int = 100, tqdm=lambda x: x) -> torch.Tensor:
        """
        音声生成を行うメソッド.

        Args:
            condition (torch.Tensor): 条件付け特徴量
            num_time_steps (int, optional): 生成するサンプルサイズ. Defaults to 100.
            tqdm (_type_, optional): プログレスバー関数. Defaults to lambdax:x.

        Returns:
            torch.Tensor: 生成した音声
        """
        outputs = []
        N = condition.shape[0]

        if tqdm is None:
            ts = range(num_time_steps)
        else:
            ts = tqdm(range(num_time_steps))

        # Local conditioning
        c = self.upsample_net(c)
        c = c.transpose(1, 2).contiguous()  # [N, T, C] -> [N, C, T]

        # 自己回帰生成における初期値
        current_input = torch.zeros(N, 1, self.out_channels).to(c.device)
        current_input[:, :, int(mulaw_quantize(0))] = 1

        # 逐次的に生成
        for t in ts:
            # 時刻tにおける入力は，時刻t-1における出力
            if t > 0:
                current_input = outputs[-1]

            # 時刻tにおける条件付け特徴量
            ct = c[:, t, :].unsqueeze(1)

            x = current_input
            x = self.first_conv.incremental_forward(x)

            skips = 0
            for f in self.main_conv_layers:
                x, h = f(x, ct)
                skips += h
            x = skips

            for f in self.last_conv_layers:
                if hasattr(f, 'incremental_forward'):
                    x = f.incremental_forward(x)
                else:
                    x = f(x)

            # Softmax関数によって，出力をカテゴリカル分布のパラメータに変換
            x = F.softmax(x.view(N, -1), dim=1)
            # カテゴリカル分布からサンプリング
            x = torch.distributions.OneHotCategorical(x).sample()
            outputs += [x.data]

        # 各時刻における出力を結合
        outputs = torch.stack(outputs)  # [T, N, C]
        # [T, N, C] -> [N, T, C] -> [N, C, T]
        outputs = outputs.transpose(0, 1).transpose(1, 2).contiguous()

        return outputs


if __name__ == '__main__':
    wavenet = WaveNet(out_channels=256, layers=2, stacks=1, kernel_size=2, cin_channels=64)

    # wavenetにおける順伝播の動作確認
    x = torch.randint(0, 255, size=(16, 16000))
    c = torch.rand(16, 64, 16000 // 80)  # フレームシフト: 80とした場合の条件付け特徴量
    print(f'入力のサイズ: {x.shape}')
    print(f'条件付け特徴量のサイズ: {c.shape}')

    # アップサンプリングの動作確認
    c_up = wavenet.upsample_net(c)
    print(f'アップサンプリング後のサイズ: {c_up.shape}')

    output = wavenet(x, c)
    print(f'出力のサイズ: {output.shape}')
