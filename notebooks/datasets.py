import csv
import os
from collections import Counter
from typing import Dict, List, Optional, Union

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BINARY_DATASETS = [
    "accidents",
    "ad",
    "baudio",
    "bbc",
    "binarized_mnist",
    "bnetflix",
    "book",
    "c20ng",
    "cr52",
    "cwebkb",
    "dna",
    "jester",
    "kdd",
    "kosarek",
    "msnbc",
    "msweb",
    "mushrooms",
    "nltcs",
    "ocr_letters",
    "plants",
    "pumsb_star",
    "tmovie",
    "tretail",
]

UCI_DATASETS = ["power", "gas", "hepmass", "miniboone", "bsds300"]

ARTIFICIAL_DATASETS = ["ring", "rings", "funnel", "banana", "cosine", "spiral"]


def load_binary_dataset(
    name: str,
    path: str = "datasets",
    sep: str = ",",
    dtype: Union[str, np.dtype] = np.int64,
    splits: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    def csv_2_numpy(filename: str, path: str, sep: str, dtype: Union[str, np.dtype]) -> np.ndarray:
        reader = csv.reader(open(os.path.join(path, filename), "r"), delimiter=sep)
        return np.array(list(reader), dtype=dtype)

    if splits is None:
        splits = ["train", "valid", "test"]
    filenames = map(lambda s: os.path.join(name, "{0}.{1}.{2}".format(name, s, "data")), splits)
    return dict(zip(splits, map(lambda fname: csv_2_numpy(fname, path, sep, dtype), filenames)))


def load_uci_dataset(
    name: str, path: str = "datasets", dtype: Union[str, np.dtype] = np.float32
) -> Dict[str, np.ndarray]:
    if name == "power":
        return load_uci_power(path=path, dtype=dtype)
    if name == "gas":
        return load_uci_gas(path=path, dtype=dtype)
    if name == "hepmass":
        return load_uci_hepmass(path=path, dtype=dtype)
    if name == "miniboone":
        return load_uci_miniboone(path=path, dtype=dtype)
    if name == "bsds300":
        return load_uci_bsds300(path=path, dtype=dtype)
    raise ValueError(f"Unknown UCI dataset called '{name}'")


def load_artificial_dataset(
    name: str,
    num_samples: int,
    valid_test_perc: float = 0.2,
    seed: int = 42,
    dtype: np.dtype = np.float32,
    discretize: bool = False,
    discretize_bins: int = 32,
    **kwargs,
) -> Dict[str, np.ndarray]:
    num_valid_samples = int(num_samples * valid_test_perc * 0.5)
    num_test_samples = int(num_samples * valid_test_perc)
    total_num_samples = num_samples + num_valid_samples + num_test_samples
    if name == "ring":
        data = sample_single_ring(total_num_samples, seed=seed, **kwargs)
    elif name == "rings":
        data = sample_nested_rings(total_num_samples, seed=seed, **kwargs)
    elif name == "funnel":
        data = sample_funnel(total_num_samples, seed=seed, **kwargs)
        data = rotate2d_samples(data)
    elif name == "banana":
        data = sample_banana(total_num_samples, seed=seed, **kwargs)
    elif name == "cosine":
        data = sample_cosine(total_num_samples, seed=seed, **kwargs)
        data = rotate2d_samples(data)
    elif name == "spiral":
        data = spiral_sample(total_num_samples, seed=seed, **kwargs)
    else:
        raise ValueError(f"Unknown artificial dataset called '{name}'")
    data = data.astype(dtype=dtype, copy=False)

    # Standardize the data
    data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-10)

    # Discretize data, if specified
    if discretize:
        xlim, ylim = (np.min(data[:, 0]), np.max(data[:, 0])), (
            np.min(data[:, 1]),
            np.max(data[:, 1]),
        )
        _, xedges, yedges = np.histogram2d(
            data[:, 0], data[:, 1], bins=discretize_bins, range=[xlim, ylim]
        )
        quantized_xdata = np.searchsorted(xedges[:-1], data[:, 0], side="right") - 1
        quantized_ydata = np.searchsorted(yedges[:-1], data[:, 1], side="right") - 1
        data = np.stack([quantized_xdata, quantized_ydata], axis=1)

    # Split the data
    data_train, data_valid_test = train_test_split(
        data, test_size=num_valid_samples + num_test_samples, shuffle=True, random_state=seed
    )
    data_valid, data_test = train_test_split(
        data_valid_test, test_size=num_test_samples, shuffle=False
    )
    return dict(train=data_train, valid=data_valid, test=data_test)


def load_uci_power(
    path: str = "datasets", dtype: Union[str, np.dtype] = np.float32
) -> Dict[str, np.ndarray]:
    data = np.load(os.path.join(path, "power", "data.npy"))
    rng = np.random.RandomState(42)
    rng.shuffle(data)

    # Clean data and add noise
    data = np.delete(data, 3, axis=1)
    data = np.delete(data, 1, axis=1)
    voltage_noise = 0.01 * rng.rand(len(data), 1)
    gap_noise = 0.001 * rng.rand(len(data), 1)
    sm_noise = rng.rand(len(data), 3)
    time_noise = np.zeros((len(data), 1))
    noise = np.hstack((gap_noise, voltage_noise, sm_noise, time_noise))
    data = (data + noise).astype(dtype, copy=False)

    # Split data
    num_test_samples = int(0.1 * len(data))
    data_test = data[-num_test_samples:]
    data = data[0:-num_test_samples]
    num_valid_samples = int(0.1 * len(data))
    data_valid = data[-num_valid_samples:]
    data_train = data[0:-num_valid_samples]

    # Standardize data
    data = np.vstack((data_train, data_valid))
    mu, s = np.mean(data, axis=0), np.std(data, axis=0)
    data_train = (data_train - mu) / s
    data_valid = (data_valid - mu) / s
    data_test = (data_test - mu) / s
    return dict(train=data_train, valid=data_valid, test=data_test)


def load_uci_gas(
    path: str = "datasets", dtype: Union[str, np.dtype] = np.float32
) -> Dict[str, np.ndarray]:
    def compute_correlations(data: pd.DataFrame):
        return (data.corr() > 0.98).to_numpy().sum(axis=1)

    # Load and clean data
    data = pd.read_pickle(os.path.join(path, "gas", "ethylene_CO.pickle"))
    data.drop("Meth", axis=1, inplace=True)
    data.drop("Eth", axis=1, inplace=True)
    data.drop("Time", axis=1, inplace=True)

    # Remove some highly-correlated columns and standardize
    corrs = compute_correlations(data)
    while np.any(corrs > 1):
        col_to_remove = np.where(corrs > 1)[0][0]
        col_name = data.columns[col_to_remove]
        data.drop(col_name, axis=1, inplace=True)
        corrs = compute_correlations(data)
    data = (data - data.mean()) / data.std()
    data = data.to_numpy().astype(dtype, copy=False)

    # Split data
    num_test_samples = int(0.1 * data.shape[0])
    data_test = data[-num_test_samples:]
    data_train = data[0:-num_test_samples]
    num_valid_samples = int(0.1 * len(data_train))
    data_valid = data_train[-num_valid_samples:]
    data_train = data_train[0:-num_valid_samples]

    return dict(train=data_train, valid=data_valid, test=data_test)


def load_uci_hepmass(
    path: str = "datasets", dtype: Union[str, np.dtype] = np.float32
) -> Dict[str, np.ndarray]:
    # Load the data
    data_train = pd.read_csv(os.path.join(path, "hepmass", "1000_train.csv"), index_col=False)
    data_test = pd.read_csv(os.path.join(path, "hepmass", "1000_test.csv"), index_col=False)

    # Gets rid of any background noise examples i.e. class label 0
    data_train = data_train[data_train[data_train.columns[0]] == 1]
    data_train = data_train.drop(data_train.columns[0], axis=1)
    data_test = data_test[data_test[data_test.columns[0]] == 1]
    data_test = data_test.drop(data_test.columns[0], axis=1)
    data_test = data_test.drop(data_test.columns[-1], axis=1)

    # Standardize the data
    mu, s = np.mean(data_train, axis=0), np.std(data_train, axis=0)
    data_train = (data_train - mu) / s
    data_test = (data_test - mu) / s
    data_train, data_test = data_train.to_numpy(), data_test.to_numpy()
    data_train = data_train.astype(dtype, copy=False)
    data_test = data_test.astype(dtype, copy=False)

    # Remove any features that have too many re-occurring real values.
    i, features_to_remove = 0, []
    for feature in data_train.T:
        c = Counter(feature)
        max_count = np.array([v for k, v in sorted(c.items())])[0]
        if max_count > 5:
            features_to_remove.append(i)
        i += 1
    data_train = data_train[
        :, [i for i in range(data_train.shape[1]) if i not in features_to_remove]
    ]
    data_test = data_test[:, [i for i in range(data_test.shape[1]) if i not in features_to_remove]]

    # Split the training data
    num_valid = int(0.1 * len(data_train))
    data_valid = data_train[-num_valid:]
    data_train = data_train[0:-num_valid]

    return dict(train=data_train, valid=data_valid, test=data_test)


def load_uci_miniboone(
    path: str = "datasets", dtype: Union[str, np.dtype] = np.float32
) -> Dict[str, np.ndarray]:
    # Load and split the data
    data = np.load(os.path.join(path, "miniboone", "data.npy"))
    data = data.astype(dtype=dtype, copy=False)
    num_test_samples = int(0.1 * len(data))
    data_test = data[-num_test_samples:]
    data = data[0:-num_test_samples]
    num_valid_samples = int(0.1 * len(data))
    data_valid = data[-num_valid_samples:]
    data_train = data[0:-num_test_samples]

    # Standardize the data
    data = np.vstack((data_train, data_valid))
    mu, s = np.mean(data, axis=0), np.std(data, axis=0)
    data_train = (data_train - mu) / s
    data_valid = (data_valid - mu) / s
    data_test = (data_test - mu) / s

    return dict(train=data_train, valid=data_valid, test=data_test)


def load_uci_bsds300(
    path: str = "datasets", dtype: Union[str, np.dtype] = np.float32
) -> Dict[str, np.ndarray]:
    f = h5py.File(os.path.join(path, "BSDS300", "BSDS300.hdf5"), "r")
    data_train = f["train"][:].astype(dtype, copy=False)
    data_valid = f["validation"][:].astype(dtype, copy=False)
    data_test = f["test"][:].astype(dtype, copy=False)
    return dict(train=data_train, valid=data_valid, test=data_test)


def sample_single_ring(
    num_samples: int, dim: int = 2, sigma: float = 0.26, seed: int = 42
) -> np.ndarray:
    return sample_rings(num_samples, dim, sigma, radia=[1], seed=seed)


def sample_nested_rings(
    num_samples: int, dim: int = 2, sigma: float = 0.2, seed: int = 42
) -> np.ndarray:
    return sample_rings(num_samples, dim, sigma, radia=[1, 3, 5], seed=seed)


def sample_funnel(num_samples: int, dim: int = 2, sigma: float = 2.0, seed: int = 42) -> np.ndarray:
    def thresh(x: np.ndarray, low_lim: float = 0.0, high_lim: float = 5.0):
        return np.clip(np.exp(-x), low_lim, high_lim)

    random_state = np.random.RandomState(seed)
    data = random_state.randn(num_samples, dim)
    data[:, 0] *= sigma
    v = thresh(data[:, 0:1])
    data[:, 1:] = data[:, 1:] * np.sqrt(v)
    return data


def sample_banana(
    num_samples: int, dim: int = 2, sigma: float = 2.0, cf: float = 0.2, seed: int = 42
) -> np.ndarray:
    random_state = np.random.RandomState(seed)
    data = random_state.randn(num_samples, dim)
    data[:, 0] = sigma * data[:, 0]
    data[:, 1] = data[:, 1] + cf * (data[:, 0] ** 2 - sigma**2)
    if dim > 2:
        data[:, 2:] = random_state.randn(num_samples, dim - 2)
    return data


def sample_cosine(
    num_samples: int,
    dim: int = 2,
    sigma: float = 1.0,
    xlim: float = 4.0,
    omega: float = 2.0,
    alpha: float = 3.0,
    seed: int = 42,
) -> np.ndarray:
    random_state = np.random.RandomState(seed)
    x0 = random_state.uniform(-xlim, xlim, num_samples)
    x1 = alpha * np.cos(omega * x0)
    x = random_state.randn(num_samples, dim)
    x[:, 0] = x0
    x[:, 1] = sigma * x[:, 1] + x1
    return x


def spiral_sample(
    num_samples: int,
    dim: int = 2,
    sigma: float = 0.5,
    eps: float = 1.0,
    r_scale: float = 1.5,
    length: float = np.pi,
    starts: Optional[list] = None,
    seed: int = 42,
) -> np.ndarray:
    if starts is None:
        starts = [0.0, 2.0 / 3, 4.0 / 3]
    starts = length * np.asarray(starts)
    nstart = len(starts)

    random_state = np.random.RandomState(seed)
    data = np.zeros((num_samples + nstart, dim))
    batch_size = np.floor_divide(num_samples + nstart, nstart)

    def branch_params(a: np.ndarray, st: float):
        n = len(a)
        a = length * (a ** (1.0 / eps)) + st
        r = (a - st) * r_scale
        m = np.zeros((n, dim))
        v = np.ones((n, dim)) * sigma
        m[:, 0] = r * np.cos(a)
        m[:, 1] = r * np.sin(a)
        v[:, :2] = (a[:, None] - st) / length * sigma + 0.1
        return m, v

    def sample_branch(n: int, st: float):
        a = random_state.uniform(0, 1, n)
        m, v = branch_params(a, st)
        return m + np.random.randn(n, dim) * v

    for si, s in enumerate(starts):
        data[si * batch_size : (si + 1) * batch_size] = sample_branch(batch_size, s)
    return data[:num_samples]


def rotate2d_samples(data: np.ndarray, radia: float = np.pi * 0.25) -> np.ndarray:
    rot_data = data.copy()
    ox, oy = np.mean(data, axis=0)
    rot_data[:, 0] = ox + np.cos(radia) * (data[:, 0] - ox) - np.sin(radia) * (data[:, 1] - oy)
    rot_data[:, 1] = oy + np.sin(radia) * (data[:, 0] - ox) + np.cos(radia) * (data[:, 1] - oy)
    return rot_data


def sample_rings(
    num_samples: int, dim: int, sigma: float = 0.1, radia: Optional[list] = None, seed: int = 42
):
    assert dim >= 2
    if radia is None:
        radia = [1, 3, 5]

    random_state = np.random.RandomState(seed)
    radia = np.asarray(radia)
    angles = random_state.rand(num_samples) * 2 * np.pi
    noise = random_state.randn(num_samples) * sigma

    weights = 2 * np.pi * radia
    weights /= np.sum(weights)

    radia_inds = random_state.choice(len(radia), num_samples, p=weights)
    radius_samples = radia[radia_inds] + noise

    xs = radius_samples * np.sin(angles)
    ys = radius_samples * np.cos(angles)
    x = np.vstack((xs, ys)).T.reshape(num_samples, 2)

    result = np.zeros((num_samples, dim))
    result[:, :2] = x
    if dim > 2:
        result[:, 2:] = random_state.randn(num_samples, dim - 2) * sigma
    return result
