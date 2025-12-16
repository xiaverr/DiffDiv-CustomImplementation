import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from datasets import YooChooseDataset 
from datasets import load_yoochoose_data 
from torch.utils.data import DataLoader
from model import DiffDiv
from tqdm import tqdm
import json
from datetime import datetime
import csv
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler 
import os
import re

# 超参数（可根据 YooChoose 调整）
class Config:
    item_size = 0  # 将在数据加载后设置
    embed_dim = 128
    hidden_dim = 16
    latent_dim = 64
    max_seq_len = 5 #15
    T = 20           # diffusion steps
    beta_min = 1e-4
    beta_max = 0.02
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    weight_dir = "./weight"
    epochs = 50

def evaluate(epoch, config, model, dataloader, k=20, device='cuda'):
    model.eval()
    recalls, mrrs = [], []


    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for seq, target in pbar:
            seq, target = seq.to(config.device), target.to(config.device)
            #print("target sample:", target[:5]) 
            # 获取全 item logits
            logits = model(seq, train=False)   # [B, num_items]

            # 排除 padding item（通常 item_id 从 1 开始）
            # 如果你的 target 是 0-indexed（含 padding），需 mask
            # 这里假设 target ∈ [1, num_items]
            topk = logits.topk(k, dim=1).indices  # 如果 logits 对应 item 1~N，而索引是 0~N-1

            # 转为 CPU 比较
            topk = topk.cpu()
            target = target.cpu().unsqueeze(1)  # [B, 1]

            # Recall@k
            hit = (topk == target).any(dim=1).float()
            recalls.append(hit.mean().item())

            # MRR@k
            rank = (topk == target).nonzero(as_tuple=True)[1] + 1  # rank 从 1 开始
            mrr = (1.0 / rank.float()).mean().item() if len(rank) > 0 else 0.0
            mrrs.append(mrr)

    return {
        'recall': sum(recalls) / len(recalls),
        'mrr': sum(mrrs) / len(mrrs)
    }

def save_checkpoint(model, epoch, save_dir="./weight"):
    """
    保存模型权重到指定目录。
    """
    filename = f"diffdiv_epoch_{epoch:02d}.pth"
    filepath = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), filepath)
    print(f"✅ Model saved to {filepath}")

def save_config(config, path):
    if hasattr(config, '__dict__'):
        config_dict = vars(config)
    else:
        config_dict = config
    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=4)

# 使用示例
if __name__ == "__main__":

    config = Config()

    print(config.device.type)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 1. 加载并预处理数据
    sessions, item2id, id2item, num_items = load_yoochoose_data()

    train_sessions, val_sessions = train_test_split(
        sessions,
        test_size=0.1,          # 10% 作为验证集
        random_state=42         # 可复现
    )

    print(f"Train sessions: {len(train_sessions)}")
    print(f"Val sessions:   {len(val_sessions)}")
    
    # 创建数据集
    train_dataset = YooChooseDataset(train_sessions, max_seq_len=20, name="Train")
    val_dataset = YooChooseDataset(val_sessions, max_seq_len=20, name="Val")   # ← 验证集 dataset


    train_dataloader = DataLoader(train_dataset, batch_size=2048, shuffle=True, num_workers=4, pin_memory=True)

    val_dataloader = DataLoader(val_dataset, batch_size=4096, shuffle=True, num_workers=4, pin_memory=True)
    
    # 3. 打印样例
    for seq, target in train_dataloader:
        print("seq shape:", seq.shape)      # [B, 20]
        print("target shape:", target.shape) # [B]
        print("Sample seq:", seq[0])
        print("Sample target:", target[0])
        break

    
    config.item_size = num_items + 1  # +1 for padding idx 0

    model = DiffDiv(config).to(config.device)

    all_files = []
    cur_epoch = 0
    for root, dirs, filenames in os.walk(config.weight_dir):
        for filename in filenames:
            all_files.append(os.path.join(root, filename))
    if len(all_files):
        target_weight = sorted(all_files)[-1]
        model.load_state_dict(torch.load(target_weight, map_location=config.device.type))
        match = re.search(r'\d+', target_weight)
        if match:
            num_str = match.group(0)
            cur_epoch = int(num_str) + 1
    print(f"将从epoch:${cur_epoch}开始训练!")

    val_metrics = evaluate(1, config, model, val_dataloader, k=20)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    os.makedirs(config.weight_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    exp_dir = f"experiments/run_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    print(f"📁 Experiment directory: {exp_dir}")

    save_config(config, os.path.join(exp_dir, "config.json"))


    log_path = os.path.join(exp_dir, "train_log.csv")
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_recall@20", "val_mrr@20"])

    best_recall = 0.0

    use_amp = (config.device.type == 'cuda:1')
    if use_amp:
        scaler = GradScaler()
    

    for epoch in range(cur_epoch, config.epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        for seq, target in pbar:
            seq, target = seq.to(config.device), target.to(config.device)

            optimizer.zero_grad()
            with autocast():
                loss, mse, kld = model(seq, target, train=True)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()  # 更新缩放因子
    
            pbar.set_postfix({'loss': f"{loss.item():.9f}"})  # 显示 6 位小数
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_dataloader)
        save_checkpoint(model, epoch + 1)

        model.eval()
        val_metrics = evaluate(epoch, config, model, val_dataloader, k=20)
        recall20 = val_metrics['recall']
        mrr20 = val_metrics['mrr']

         # --- 保存最佳模型 ---
        if recall20 > best_recall:
            best_recall = recall20
            torch.save(model.state_dict(), os.path.join(exp_dir, "model_best.pt"))

        # --- 写入日志 ---
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, f"{avg_train_loss:.9f}", f"{recall20:.9f}", f"{mrr20:.9f}"])

        print(f"✅ Epoch {epoch+1} | Train Loss: {avg_train_loss:.9f} | "
            f"Val Recall@20: {recall20:.9f} | Val MRR@20: {mrr20:.9f}")

        
            

    