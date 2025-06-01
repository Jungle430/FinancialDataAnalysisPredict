from concurrent.futures import ThreadPoolExecutor
import torch
from torch import Tensor
from torch.nn import (
    MSELoss,
    Module,
    Linear,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from torch.optim import Optimizer, Adam
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter  # 引入SummaryWriter用于TensorBoard
from loguru import logger

import os
import pandas as pd
import numpy as np
import pickle  # 导入 pickle 模块，用于保存和加载 scaler
from sklearn.preprocessing import MinMaxScaler  # 引入MinMaxScaler用于数据归一化
from typing import Tuple
from constData import DEFAULT_WEEK_WINDOW_SIZE
from db import STOCK_VALUES, query_stock_all_code, query_stock_data_by_code, connection
from device import get_device
from threadPoolUtil import get_transformer_thread_pool


class StockDataSet(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        code: str,
        window_size: int = DEFAULT_WEEK_WINDOW_SIZE,
        is_train: bool = True,
    ) -> None:
        self.df = df
        self.code = code
        self.window_size = window_size
        self.is_train = is_train
        data, _ = self.load_data()
        self.data = torch.tensor(data, dtype=torch.float32)

    def load_data(self) -> Tuple[np.ndarray, MinMaxScaler]:
        # 确保第一列是数值类型，并赋值为0~len-1
        # 注意：如果第一列是日期/时间戳等，可能不适合作为数值特征直接参与模型训练
        # 如果它仅仅是索引，可以考虑在归一化之前将其分离或进行特殊处理
        # 这里假设它需要作为一个特征参与归一化

        # 将第一列（时间戳）转换为序号
        self.df.iloc[:, 0] = range(len(self.df))  # type: ignore

        # 将所有列明确转换为数值类型,转换不了的变成NaN
        for col in self.df.columns:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # 处理NaN值，例如填充0
        self.df = self.df.fillna(0)

        # === 数据归一化和 scaler 保存/加载 ===
        scaler_filepath = (
            f"model/scaler_{self.code}.pkl"  # 修改为 .pkl 扩展名，使用 pickle 保存
        )

        if self.is_train:
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(self.df.values)
            # 使用 pickle 保存 scaler
            with open(scaler_filepath, "wb") as f:
                pickle.dump(scaler, f)
        else:
            # 尝试加载之前保存的 scaler
            if not os.path.exists(scaler_filepath):
                raise FileNotFoundError(
                    f"Scaler file not found at {scaler_filepath}! Please ensure you have trained the model and saved the scaler."
                )

            # 使用 pickle 加载 scaler
            with open(scaler_filepath, "rb") as f:
                scaler: MinMaxScaler = pickle.load(f)
            normalized_data = scaler.transform(self.df.values)

        return normalized_data, scaler

    def __len__(self) -> int:
        # 每个样本是 window_size 个数据，预测下一个
        return len(self.data) - self.window_size

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        # x 和 y 已经是 float32 的 torch.tensor
        x = self.data[idx : idx + self.window_size]  # 1 ~ 7
        y = self.data[idx + self.window_size]  # 8
        return x, y


class TransformerPredictor(Module):
    def __init__(
        self, feature_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2
    ) -> None:
        super().__init__()
        self.input_proj = Linear(feature_dim, d_model)
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )  # 这个意思就是num_layers层的encoder_layer
        self.output_proj = Linear(d_model, feature_dim)

    def forward(self, x: Tensor) -> Tensor:
        # x: [batch_size, window_size, feature_dim]
        x = self.input_proj(x)  # [batch_size, window_size, d_model]
        x = self.transformer_encoder(x)  # [batch_size, window_size, d_model]
        out = self.output_proj(x[:, -1, :])  # 只用最后一个时间步做预测
        return out  # [batch_size, feature_dim]


def train_one_epoch(
    model: Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: MSELoss,
    device: torch.device = torch.device("cpu"),
    grad_clip_norm: float = 1.0,
) -> float:
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    avg_loss = total_loss / len(dataloader.dataset)  # type: ignore
    return avg_loss


def evaluate(
    model: Module,
    dataloader: DataLoader,
    criterion: MSELoss,
    device: torch.device = torch.device("cpu"),
) -> float:
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            total_loss += loss.item() * x.size(0)
    avg_loss = total_loss / len(dataloader.dataset)  # type: ignore
    return avg_loss


def predict(
    model_path: str,
    data: pd.DataFrame,
    code: str,
    feature_dim: int = len(STOCK_VALUES.split(", ")),
    window_size: int = DEFAULT_WEEK_WINDOW_SIZE,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """
    加载训练好的模型并对新数据进行预测。

    Args:
        model_path (str): 保存的模型参数文件路径(.pth)
        data (pd.DataFrame): 最新数据
        code (str): 股票代码
        window_size (int): 输入序列的窗口大小
        feature_dim (int): 输入特征的维度
        device (torch.device): 预测时使用的设备

    Returns:
        numpy.ndarray: 预测结果（反归一化后的原始数据范围）。
    """
    # 确保模型保存目录存在
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # 加载 scaler 对象 (使用 pickle)
    scaler_path = os.path.join(model_dir, f"scaler_{code}.pkl")  # 注意扩展名
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler file not found at {scaler_path}. Please ensure it was saved during training."
        )

    with open(scaler_path, "rb") as f:
        scaler: MinMaxScaler = pickle.load(f)

    # 1. 准备预测数据
    df_raw = data.copy()

    # 对数据进行相同的预处理 (赋值第一列, 类型转换, 填充NaN)
    df_raw.iloc[:, 0] = range(len(df_raw))  # type: ignore
    for col in df_raw.columns:
        df_raw[col] = pd.to_numeric(
            df_raw[col], errors="coerce"
        )  # bug fix: use df[col] instead of df_raw[col] here if it is for inplace update of df_raw
    df_raw = df_raw.fillna(0)

    # 取最后 window_size 行作为输入
    if len(df_raw) < window_size:
        raise ValueError(
            f"Input data must have at least {window_size} rows for prediction."
        )
    input_data_raw = df_raw.tail(window_size).values
    # 使用训练时fit的scaler进行归一化
    input_data_normalized = scaler.transform(input_data_raw)

    # 转换为 PyTorch Tensor，并添加 batch 维度
    input_tensor = (
        torch.tensor(input_data_normalized, dtype=torch.float32).unsqueeze(0).to(device)
    )  # [1, window_size, feature_dim]

    # 2. 加载模型
    model = TransformerPredictor(feature_dim=feature_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 设置为评估模式

    # 3. 进行预测
    with torch.no_grad():
        prediction_normalized = model(input_tensor).cpu().numpy()  # [1, feature_dim]

    # 4. 反归一化预测结果
    dummy_row_for_inverse = np.zeros((1, scaler.n_features_in_))
    dummy_row_for_inverse[0, :] = prediction_normalized[0, :]

    prediction_original_scale = scaler.inverse_transform(dummy_row_for_inverse)
    return prediction_original_scale[0]  # 返回第一行（预测结果）


def train_and_save_model(
    td_conn, code: str, num_epochs: int = 60, grad_clip_norm: float = 1.0
):
    # 初始化 TensorBoard SummaryWriter
    writer = SummaryWriter("runs/btc_transformer_experiment")
    df = query_stock_data_by_code(conn=td_conn, code=code)
    train_dataset = StockDataSet(
        df=df, code=code, is_train=True
    )  # 训练时 is_train=True

    # 随机分割数据集
    train_size = int(len(train_dataset) * 0.8)
    test_size = len(train_dataset) - train_size
    train_subset, test_subset = random_split(train_dataset, [train_size, test_size])

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

    device = get_device()
    logger.info(f"code={code}, Using device: {device}")
    feature_dim = len(STOCK_VALUES.split(", "))  # 从数据集中获取特征维度
    model = TransformerPredictor(feature_dim=feature_dim).to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)  # 尝试更小的学习率
    criterion = MSELoss()

    logger.info("--- Starting training ---")
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, grad_clip_norm
        )
        val_loss = evaluate(model, test_loader, criterion, device)
        # 记录到 TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
    logger.info("--- Training complete ---")

    # 关闭 SummaryWriter
    writer.close()

    # 保存模型参数
    torch.save(model.state_dict(), f"./model/transformer_predictor_{code}.pth")


def train_model_task(
    td_conn,
    thread_pool: ThreadPoolExecutor,
    num_epochs: int = 60,
    grad_clip_norm: float = 1.0,
):
    codes = query_stock_all_code(conn=td_conn)
    for code in codes:
        thread_pool.submit(
            train_and_save_model, td_conn, code, num_epochs, grad_clip_norm
        )
