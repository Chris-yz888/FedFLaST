"""
CIFAR-10-C / CIFAR-100-C Dataset Loader
基于 Hendrycks & Dietterich (ICLR 2019) "Benchmarking Neural Network Robustness
to Common Corruptions and Perturbations"

数据格式: .npy 文件
  - data.npy: shape (10000, 5, 15, 32, 32, 3) 或 (5, 15, 10000, 32, 32, 3)
    维度含义: [N_samples, severity(5), corruption_type(15), H, W, C]
  - labels.npy: shape (10000,)

15 种腐蚀类型:
  0: gaussian_noise      1: shot_noise         2: impulse_noise
  3: defocus_blur        4: glass_blur         5: motion_blur
  6: zoom_blur           7: snow               8: frost
  9: fog                10: brightness        11: contrast
 12: elastic_transform  13: pixelate          14: jpeg_compression
"""

import logging
import numpy as np
import torch
import torch.utils.data as data

logger = logging.getLogger(__name__)

# 15 corruption types from Hendrycks & Dietterich 2019
CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur',
    'zoom_blur', 'snow', 'frost',
    'fog', 'brightness', 'contrast',
    'elastic_transform', 'pixelate', 'jpeg_compression',
]


class CIFAR10C(data.Dataset):
    """
    CIFAR-10-C 数据集类。
    支持按腐蚀类型 (corruption) 和严重程度 (severity) 筛选数据。
    """

    def __init__(self, root, corruption=None, severity=None, transform=None):
        """
        Args:
            root: 数据目录路径，包含 data.npy 和 labels.npy
            corruption: 腐蚀类型，int (0-14) 或 str。None 表示所有类型。
            severity: 严重程度，int (1-5)。None 表示所有级别。
            transform: 图像变换 (torchvision transforms)
        """
        self.root = root
        self.transform = transform

        # 加载数据
        data_path = f"{root}/data.npy"
        labels_path = f"{root}/labels.npy"

        raw_data = np.load(data_path)       # (10000, 5, 15, 32, 32, 3)
        self.labels = np.load(labels_path)  # (10000,)

        # 处理不同数据形状: 统一为 (N, H, W, C)
        # 原始形状可能是 (5, 15, 10000, 32, 32, 3) 或 (10000, 5, 15, 32, 32, 3)
        if raw_data.shape[0] == 5 and raw_data.shape[1] == 15:
            # shape: (5, 15, 10000, 32, 32, 3) -> (10000, 5, 15, 32, 32, 3)
            raw_data = np.transpose(raw_data, (2, 0, 1, 3, 4, 5))
        elif raw_data.shape[0] == 10000 and raw_data.shape[1] == 5 and raw_data.shape[2] == 15:
            pass  # shape already (10000, 5, 15, 32, 32, 3)
        else:
            raise ValueError(
                f"Unexpected data shape: {raw_data.shape}. "
                f"Expected (10000, 5, 15, 32, 32, 3) or (5, 15, 10000, 32, 32, 3)"
            )

        # 筛选: 只保留指定的 corruption 和 severity
        self.data_raw = raw_data  # (10000, 5, 15, 32, 32, 3)
        self.n_samples = 10000
        self.n_severities = 5
        self.n_corruptions = 15

        # 确定要评估的 (severity, corruption) 组合
        if corruption is not None:
            if isinstance(corruption, str):
                corruption = CORRUPTIONS.index(corruption.lower())
            self.corruptions_subset = [corruption]
        else:
            self.corruptions_subset = list(range(15))

        if severity is not None:
            self.severities_subset = [severity - 1]  # 转换为 0-indexed
        else:
            self.severities_subset = list(range(5))

        # 展平: [N, severity, corruption, H, W, C] -> [N * len(sev) * len(corr), H, W, C]
        self._flatten_data()

        logger.info(f"CIFAR-10-C loaded: {len(self)} images "
                     f"(corruptions={self.corruptions_subset}, severities={[s+1 for s in self.severities_subset]})")

    def _flatten_data(self):
        """将选定的 (severity, corruption) 组合展平为 (M, H, W, C)"""
        images = []
        targets = []
        for s_idx in self.severities_subset:
            for c_idx in self.corruptions_subset:
                # 取所有样本在该 severity 和 corruption 下的图像
                subset = self.data_raw[:, s_idx, c_idx, :, :, :]  # (10000, 32, 32, 3)
                images.append(subset)
                targets.append(self.labels.copy())
        self.data = np.concatenate(images, axis=0)  # (M, 32, 32, 3)
        self.targets = np.concatenate(targets, axis=0)  # (M,)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = img.astype(np.uint8)  # 确保 uint8 格式

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)


def get_cifar10c_dataloader(data_dir, corruption=None, severity=None,
                            batch_size=128, num_workers=4, transform=None):
    """
    创建 CIFAR-10-C DataLoader。

    Args:
        data_dir: CIFAR-10-C 数据目录 (包含 data.npy 和 labels.npy)
        corruption: 腐蚀类型 int/str/None (None = 所有类型)
        severity: 严重程度 int 1-5/None (None = 所有级别)
        batch_size: 批量大小
        num_workers: 数据加载线程数
        transform: torchvision transforms (默认使用 CIFAR-10 标准化)

    Returns:
        DataLoader
    """
    if transform is None:
        from torchvision import transforms
        CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
        CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

    dataset = CIFAR10C(data_dir, corruption=corruption, severity=severity, transform=transform)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=True)
    return dataloader


def get_all_dataloaders(data_dir, batch_size=128, num_workers=4):
    """
    创建所有 75 个 (15 corruption × 5 severity) DataLoader 的字典。

    Returns:
        dict: {(corruption_name, severity): DataLoader}
    """
    from torchvision import transforms
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    # 加载完整数据
    full_dataset = CIFAR10C(data_dir, transform=transform)
    full_data = full_dataset.data_raw  # (10000, 5, 15, 32, 32, 3)
    labels = full_dataset.labels      # (10000,)

    dataloaders = {}
    for c_idx in range(15):
        for s_idx in range(5):
            subset_data = full_data[:, s_idx, c_idx, :, :, :]  # (10000, 32, 32, 3)

            # 创建临时 Dataset
            class SubsetDataset(data.Dataset):
                def __init__(self, images, labels, transform):
                    self.images = images.astype(np.uint8)
                    self.labels = labels
                    self.transform = transform

                def __getitem__(self, idx):
                    img = self.transform(self.images[idx])
                    return img, self.labels[idx]

                def __len__(self):
                    return len(self.images)

            dataset = SubsetDataset(subset_data, labels, transform)
            dl = data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True)
            key = (CORRUPTIONS[c_idx], s_idx + 1)
            dataloaders[key] = dl

    return dataloaders
