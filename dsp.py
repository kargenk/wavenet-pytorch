import numpy as np


def mulaw(x: np.ndarray, mu: int = 256) -> np.ndarray:
    """
    Mu-Law圧縮.

    Args:
        x (np.ndarray): 入力信号
        mu (int, optional): 量子化数. Defaults to 256.

    Returns:
        np.ndarray: 圧縮された信号
    """
    return np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)


def quantize(y: np.ndarray, mu: int = 256, offset: int = 1) -> np.ndarray:
    """
    入力信号の量子化.

    Args:
        y (np.ndarray): 入力信号
        mu (int, optional): 量子化数. Defaults to 256.
        offset (int, optional): オフセット. Defaults to 1.

    Returns:
        np.ndarray: 量子化された信号
    """
    # [-1, 1] -> [0, 2] -> [0, 1] -> [0, mu]
    return ((y + offset) / 2 * mu).astype(np.int64)


def mulaw_quantize(x: np.ndarray, mu: int = 256) -> np.ndarray:
    """
    Mu-Lawアルゴリズム.

    Args:
        x (np.ndarray): 入力信号
        mu (int, optional): 量子化数. Defaults to 256.

    Returns:
        np.ndarray: μ-lawアルゴリズムを用いて圧縮した信号
    """
    return quantize(mulaw(x, mu), mu)


def inv_mulaw(y: np.ndarray, mu: int = 256) -> np.ndarray:
    """
    mu-law圧縮の逆変換.

    Args:
        y (np.ndarray): 入力信号
        mu (int, optional): 量子化数. Defaults to 256.

    Returns:
        np.ndarray: 復元した信号
    """
    return np.sign(y) * (1.0 / mu) * ((1.0 + mu) ** np.abs(y) - 1.0)


def inv_quantize(y: np.ndarray, mu: int = 256) -> np.ndarray:
    """
    量子化の逆変換.

    Args:
        y (np.ndarray): 入力信号
        mu (int, optional): 量子化数. Defaults to 256.

    Returns:
        np.ndarray: 量子化された入力を復元した信号
    """
    # [0, mu] -> [0, 2mu] -> [0, 2] -> [-1, 1]
    return 2 * y.astype(np.float32) / mu - 1


def inv_mulaw_quantize(y: np.ndarray, mu: int = 256) -> np.ndarray:
    """
    Mu-Lawアルゴリズムの逆変換.

    Args:
        y (np.ndarray): 入力信号
        mu (int, optional): 量子化数. Defaults to 256.

    Returns:
        np.ndarray: μ-law逆変換で復元した信号
    """
    return inv_mulaw(inv_quantize(y, mu), mu)
