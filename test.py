import os
from PIL import Image

input_dir = "/Users/openg/sd-lora-style/output/blur_to_clear/"
output_path = "/Users/openg/sd-lora-style/output/blur_to_clear/merged.png"

# 读取所有图片文件，按文件名排序
files = sorted([f for f in os.listdir(input_dir) if f.endswith(".png")])

# 读第一张确定尺寸
img0 = Image.open(os.path.join(input_dir, files[0]))
w, h = img0.size

cols = 5  # 列数
rows = (len(files) + cols - 1) // cols  # 计算行数

# 创建空白大图
merged_img = Image.new('RGB', (w * cols, h * rows))

for idx, filename in enumerate(files):
    img = Image.open(os.path.join(input_dir, filename))
    x = (idx % cols) * w
    y = (idx // cols) * h
    merged_img.paste(img, (x, y))

merged_img.save(output_path)
print(f"合成图已保存到: {output_path}")
