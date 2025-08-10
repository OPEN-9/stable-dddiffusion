import os
from diffusers import StableDiffusionPipeline
import torch

# 你的 LoRA 模型映射（示例）
lora_map = {
    "kodak": "./lora/kodak.safetensors",
    # "fuji": "./lora/fuji.safetensors",  # 你可以扩展其他 LoRA
}

# 基础模型
base_model = "stabilityai/stable-diffusion-2-1"

# 初始化 pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    base_model,
    torch_dtype=torch.float32,
    safety_checker=None,
).to("mps")  # Mac Studio 的 Metal 加速

pipe.set_progress_bar_config(disable=False)

def generate_image(prompt, lora_name, output_path, num_inference_steps=30, guidance_scale=10):
    if lora_name not in lora_map:
        raise ValueError(f"LoRA '{lora_name}' not found.")

    lora_path = lora_map[lora_name]

    # 加载 LoRA 权重，prefix=None 避免警告
    pipe.load_lora_weights(
        os.path.dirname(lora_path),
        weight_name=os.path.basename(lora_path),
        prefix=None
    )

    # 生成图像
    image = pipe(
        prompt=prompt,
        guidance_scale=guidance_scale,
        strength=1,
        num_inference_steps=num_inference_steps,
        height=512,
        width=512
    ).images[0]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)
    print(f"✅ Saved: {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Film grain still image of a yellow train engine sitting on top of a track,outdoors,sky,day,english text,blue sky,no humans,ground vehicle,scenery,motor vehicle , cinematic look, film look, filmic, contrast, detailed, high quality, sharp image, film color, Kodak Motion Picture Film style, different color, different people, different look, different style, 35MM Film, 16MM Film, Photographic film, music video style, artistic style, cinematic style, film granularity, film noise, image noise, artistic effect, Fujicolor, Fuji film, Analog photography, movie style, movie still, Film grain overlay, Film Grain style")
    parser.add_argument("--style", type=str, default="kodak") #lora
    parser.add_argument("--output", type=str, default="output/generated_image3.jpg") #路径
    parser.add_argument("--steps", type=int, default=30) # 生成图像的推理步数
    parser.add_argument("--scale", type=float, default=10) # 风格自然过渡的指导强度
    args = parser.parse_args()

    generate_image(args.prompt, args.style, args.output, args.steps, args.scale)
