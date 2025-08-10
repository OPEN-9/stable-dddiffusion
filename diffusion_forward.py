import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 确保输出目录存在
os.makedirs("output", exist_ok=True)

# 读取本地图片（请改成你的路径）
img_path = "/Users/openg/sd-lora-style/input/image4.jpg"
img = Image.open(img_path).convert("RGB")
img = img.resize((64, 64))
x0 = np.array(img) / 255.0

# 正向扩散（逐步加噪）
def forward_diffusion(x0, steps=8, noise_scale=0.1):
    images = [x0]
    x = x0.copy()
    for i in range(steps):
        noise = np.random.randn(*x.shape) * noise_scale
        x = np.clip(x + noise, 0, 1)
        images.append(x)
    return images

# 运行正向扩散
forward_imgs = forward_diffusion(x0, steps=6, noise_scale=0.15)

fig, axes = plt.subplots(1, len(forward_imgs), figsize=(15, 3))
for i, img in enumerate(forward_imgs):
    axes[i].imshow(img)
    axes[i].set_title(f"t={i}")
    axes[i].axis('off')

plt.tight_layout()

output_path = os.path.join("output", "forward_diffusion.png")
plt.savefig(output_path)
plt.close()