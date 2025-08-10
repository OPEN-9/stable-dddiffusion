import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# 设备设置，优先用Apple Silicon GPU (MPS)，否则CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

# 1. 改进的UNet网络（支持更高分辨率）
class ImprovedUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 下采样路径
        self.down1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # 中间层
        self.mid = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
        # 上采样路径
        self.up1 = nn.Sequential(
            nn.Conv2d(384, 128, 3, padding=1),  # 256 + 128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(192, 64, 3, padding=1),  # 128 + 64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # 输出层
        self.out = nn.Conv2d(64, 1, 1)
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )

    def forward(self, x, t):
        # 时间嵌入
        t_emb = self.time_embed(t.unsqueeze(-1))
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)  # [B, 256, 1, 1]
        
        # 下采样
        x1 = self.down1(x)
        x2 = self.down2(x1)
        
        # 中间层
        xm = self.mid(x2)
        
        # 添加时间信息
        xm = xm + t_emb
        
        # 上采样
        xu1 = self.up1(torch.cat([xm, x2], dim=1))
        xu2 = self.up2(torch.cat([xu1, x1], dim=1))
        
        return self.out(xu2)

# 2. 噪声调度参数
T = 1000
beta = torch.linspace(1e-4, 0.02, T).to(device)
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

# 3. 给x0和t添加噪声
def q_sample(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_alpha_bar = alpha_bar[t].sqrt().view(-1,1,1,1)
    sqrt_one_minus_alpha_bar = (1 - alpha_bar[t]).sqrt().view(-1,1,1,1)
    return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

# 4. 训练函数
def train():
    # 定义转换函数，避免使用lambda
    def scale(x):
        return (x - 0.5) * 2
    
    transform = transforms.Compose([
        transforms.Resize(64),  # 为了训练速度，暂时使用64x64
        transforms.ToTensor(),
        transforms.Lambda(scale)  # 使用预定义的函数
    ])
    
    dataset = torchvision.datasets.MNIST(
        root="./data", 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # 在Mac上使用num_workers=0避免多进程问题
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    model = ImprovedUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    epochs = 10  # 减少epoch数量以加快测试
    
    print("Starting training...")
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x0, _) in enumerate(dataloader):
            x0 = x0.to(device)
            batch_size = x0.size(0)
            
            # 随机时间步
            t = torch.randint(0, T, (batch_size,), device=device)
            
            # 添加噪声
            noise = torch.randn_like(x0)
            x_t = q_sample(x0, t, noise)
            
            # 预测噪声
            noise_pred = model(x_t, t.float() / T)
            loss = F.mse_loss(noise_pred, noise)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1} Batch {batch_idx} Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")
        
        # 每个epoch后保存模型
        torch.save(model.state_dict(), "diffusion_model.pth")
        print("Model saved.")
    
    return model

# 5. 采样并显示大图
@torch.no_grad()
def sample_and_plot(model, n_samples=4, n_rows=4):
    model.eval()
    # 增大生成图片的分辨率
    img_size = 128  # 从64增大到128
    x_t = torch.randn(n_samples, 1, img_size, img_size).to(device)
    saved_images = []

    # 均匀选择关键步骤 - 修正括号匹配问题
    save_steps = [int(T * (i / (n_rows - 1))) for i in range(n_rows)]  # 注意这里添加了缺失的括号
    save_steps[-1] = T - 1  # 确保最后一步是t=0

    for t in reversed(range(T)):
        t_batch = torch.full((n_samples,), t, device=device, dtype=torch.float32) / T
        noise_pred = model(x_t, t_batch)

        beta_t = beta[t]
        alpha_t = alpha[t]
        alpha_bar_t = alpha_bar[t]

        if t > 0:
            noise = torch.randn_like(x_t)
        else:
            noise = torch.zeros_like(x_t)

        x_t = (1 / alpha_t.sqrt()) * (
            x_t - ((1 - alpha_t) / (1 - alpha_bar_t).sqrt()) * noise_pred
        ) + beta_t.sqrt() * noise

        if t in save_steps:
            img = (x_t.clamp(-1, 1) + 1) / 2
            saved_images.append(img.cpu())

    # 创建大图
    plt.figure(figsize=(20, 10))  # 增大画布尺寸
    
    # 将图像网格转换为numpy数组并放大
    grid = torchvision.utils.make_grid(
        torch.cat(saved_images, dim=0),
        nrow=n_samples,
        padding=20,  # 增加间距
        pad_value=1
    )
    
    # 使用最近邻插值放大图像
    grid_np = grid.permute(1, 2, 0).numpy()
    height, width = grid_np.shape[:2]
    new_height, new_width = height*2, width*2  # 放大2倍
    
    plt.imshow(grid_np, cmap='gray', interpolation='nearest', 
              extent=[0, new_width, 0, new_height])
    plt.axis('off')
    plt.title("Diffusion Process (Noise → Clean Image)", fontsize=18, pad=20)
    plt.tight_layout()
    
    # 添加步骤标签
    for i, step in enumerate(save_steps):
        y_pos = new_height - (i * (new_height/n_rows) + new_height/(n_rows*2))
        plt.text(new_width + 50, y_pos, 
                f"Step {T-step}", 
                ha='left', va='center', fontsize=14)
    
    plt.show()

# 在主程序中调用
if __name__ == "__main__":
    # 检查是否有预训练模型
    if os.path.exists("diffusion_model.pth"):
        model = ImprovedUNet().to(device)
        model.load_state_dict(torch.load("diffusion_model.pth"))
        print("Loaded pretrained model.")
    else:
        print("No pretrained model found, training from scratch.")
        model = train()
    
    # 生成并显示大图
    sample_and_plot(model, n_samples=4, n_rows=4)
