import os
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch
from collections import Counter, defaultdict
import pickle


class YooChooseDataset(Dataset):
    def __init__(self, sessions, max_seq_len=20, name="Train"):
        """
        sessions: List[List[int]], 每个子列表是 [i1, i2, ..., iN]
        我们将构造样本：history = [i1, ..., i_{k}], target = i_{k+1}，k >= 1
        """
        self.samples = []
        self.max_seq_len = max_seq_len
        
        for sess in sessions:
            if len(sess) < 2:
                continue
            # 生成所有可能的 (history, next_item) 对
            for i in range(1, len(sess)):
                history = sess[:i]
                target = sess[i]
                # 截断长历史
                if len(history) > max_seq_len:
                    history = history[-max_seq_len:]
                self.samples.append((history, target))
        
        print(f"Total {name} samples: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        history, target = self.samples[idx]
        # padding to max_seq_len
        padded_history = [0] * (self.max_seq_len - len(history)) + history
        return torch.tensor(padded_history, dtype=torch.long), torch.tensor(target, dtype=torch.long)


def load_yoochoose_data(
    data_path="../1/yoochoose-clicks.dat",
    cache_path="data/yoochoose_processed.pkl",  # 缓存路径
    min_item_freq=5,
    min_session_length=2,
    sample_frac=None  # 可选：用于快速调试（如 0.01）
):
    """
    加载并预处理 YooChoose 数据，支持缓存加速。
    """
    # 如果缓存存在，直接加载
    if cache_path and os.path.exists(cache_path):
        print(f"📦 Loading processed data from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # ===== 否则，从头处理 =====
    print("🔄 Processing raw data (this may take a few minutes)...")

    # 1. 加载原始数据
    df = pd.read_csv(
        data_path,
        names=["session_id", "timestamp", "item_id", "category"],
        dtype={"session_id": int, "item_id": int}
    )

    # 2. 可选：采样（用于调试）
    if sample_frac is not None:
        unique_sessions = df["session_id"].unique()
        sampled_sessions = pd.Series(unique_sessions).sample(frac=sample_frac, random_state=42)
        df = df[df["session_id"].isin(sampled_sessions)]

    # 3. 过滤短会话
    session_lengths = df.groupby("session_id").size()
    valid_sessions = session_lengths[session_lengths >= min_session_length].index
    df = df[df["session_id"].isin(valid_sessions)]

    # 4. 过滤低频物品
    item_counter = Counter(df["item_id"])
    valid_items = {item for item, cnt in item_counter.items() if cnt >= min_item_freq}
    df = df[df["item_id"].isin(valid_items)]

    # 5. 构建 item2id 映射（ID 从 1 开始，0 保留给 padding）
    item2id = {item: idx + 1 for idx, item in enumerate(sorted(valid_items))}
    id2item = {idx + 1: item for item, idx in item2id.items()}
    num_items = len(item2id)

    # 6. 转换 item_id 并按 session 分组
    df["item_id"] = df["item_id"].map(item2id)
    sessions = df.groupby("session_id")["item_id"].apply(list).tolist()

    # 7. 保存缓存（确保目录存在）
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump((sessions, item2id, id2item, num_items), f)
        print(f"💾 Cached processed data to: {cache_path}")

    return sessions, item2id, id2item, num_items