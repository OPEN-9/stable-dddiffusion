import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 确保输出目录存在
os.makedirs("output", exist_ok=True)

# 读取本地图片（请改成你的路径）
img_path = "/Users/openg/sd-lora-style/output/forward_diffusion.png"
img = Image.open(img_path).convert("RGB")
img = img.resize((64, 64))
x0 = np.array(img) / 255.0  # 转成 [0,1]

# 正向扩散（逐步加噪）
def forward_diffusion(x0, steps=6, noise_scale=0.15):
    images = [x0]
    x = x0.copy()
    for i in range(steps):
        noise = np.random.randn(*x.shape) * noise_scale
        x = np.clip(x + noise, 0, 1)
        images.append(x)
    return images

# 反向扩散（逐步去噪，假设知道噪声）
def reverse_diffusion(noisy_images, noise_scale=0.15):
    images = [noisy_images[-1]]
    x = noisy_images[-1].copy()
    for _ in range(len(noisy_images)-1):
        noise = np.random.randn(*x.shape) * noise_scale
        x = np.clip(x - noise, 0, 1)  # 简单去噪
        images.append(x)
    return images

# 先生成正向加噪序列
forward_imgs = forward_diffusion(x0)

# 再生成反向去噪序列
reverse_imgs = reverse_diffusion(forward_imgs)

# 画图保存（反向去噪过程）
fig, axes = plt.subplots(1, len(reverse_imgs), figsize=(15, 3))
for i, img in enumerate(reverse_imgs):
    axes[i].imshow(img)
    axes[i].set_title(f"rev t={i}")
    axes[i].axis('off')

plt.tight_layout()

output_path = os.path.join("output", "reverse_diffusion.png")
plt.savefig(output_path)
plt.close()
