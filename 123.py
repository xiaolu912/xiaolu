import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# ========== 1. 数据加载和预处理 ==========
class CornDataset(Dataset):
    def __init__(self, file_path):
        df = pd.read_excel(file_path)

        # 假设前700列为光谱数据，后4列为目标值
        self.spectra = df.iloc[:, :700].values.astype(np.float32)
        self.targets = df.iloc[:, 700:704].values.astype(np.float32)

        # 归一化光谱数据（0-1归一化）
        self.scaler = MinMaxScaler()
        self.spectra = self.scaler.fit_transform(self.spectra)

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        return torch.tensor(self.spectra[idx]), torch.tensor(self.targets[idx])


# ========== 2. 构建模型（FNN） ==========
class FNNModel(nn.Module):
    def __init__(self):
        super(FNNModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(700, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # 输出四个成分
        )

    def forward(self, x):
        return self.model(x)


# ========== 3. 模型训练函数 ==========
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=100):
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # 验证集
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return train_losses, val_losses


# ========== 4. 模型评估 ==========
def evaluate_model(model, test_loader, device):
    model.eval()
    predictions, targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predictions.extend(outputs.cpu().numpy())
            targets.extend(y_batch.numpy())

    predictions = np.array(predictions)
    targets = np.array(targets)

    mse = mean_squared_error(targets, predictions)
    print(f"\nTest MSE: {mse:.4f}")

    # 绘图
    component_names = ['Moisture', 'Oil', 'Protein', 'Starch']
    for i in range(4):
        plt.figure()
        plt.scatter(targets[:, i], predictions[:, i], color='blue', alpha=0.7)
        plt.plot([targets[:, i].min(), targets[:, i].max()],
                 [targets[:, i].min(), targets[:, i].max()],
                 'r--', lw=2)
        plt.xlabel('True Value')
        plt.ylabel('Predicted Value')
        plt.title(f'Prediction vs True: {component_names[i]}')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ========== 5. 主程序入口 ==========
def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    dataset = CornDataset("玉米的近红外光谱数据.xlsx")

    # 数据划分（70%训练，15%验证，15%测试）
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # 模型、损失函数、优化器
    model = FNNModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # L2正则

    # 训练模型
    train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=100)

    # 评估模型
    evaluate_model(model, test_loader, device)


if __name__ == "__main__":
    main()
