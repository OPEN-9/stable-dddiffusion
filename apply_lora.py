import os
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

# LoRA 路径映射
lora_map = {
    "kodak": "./lora/kodak.safetensors",
    "kodak_portra": "./lora/kodak_portra.safetensors",
    "kodak_vision": "./lora/kodak_vision.safetensors",
}

# LoRA Prompt 推荐词
style_prompts = {
    "kodak": "cinematic photo, film grain, kodak, detailed, 35mm film, photorealistic, film look, analog style",
    "kodak_portra": "soft skin tones, cinematic look, kodak portra, warm light, 35mm, film photography",
    "kodak_vision": "vibrant color, fujifilm, fine grain, analog photo, high saturation, travel photography",
}

# 模型加载
base_model = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    base_model,
    torch_dtype=torch.float32,
    safety_checker=None,
)
pipe.enable_attention_slicing()
pipe = pipe.to("mps")

def apply_lora(image_path, lora_name, strength=0.1, guidance_scale=5.0):
    if lora_name not in lora_map:
        raise ValueError(f"LoRA '{lora_name}' not found.")
    
    # 加载 LoRA
    pipe.unload_lora_weights()
    pipe.load_lora_weights(
        os.path.dirname(lora_map[lora_name]),
        weight_name=os.path.basename(lora_map[lora_name]),
    )

    # 加载图片
    image = Image.open(image_path).convert("RGB")
    prompt = style_prompts.get(lora_name, "cinematic photo, analog style")

    # 生成
    result = pipe(
        prompt=prompt,
        image=image,
        strength=0.1,
        guidance_scale=6,
        num_inference_steps=30,
    ).images[0]

    # 保存
    os.makedirs("output", exist_ok=True)
    output_path = os.path.join("output", f"{os.path.splitext(os.path.basename(image_path))[0]}_{lora_name}.jpg")
    result.save(output_path)
    print(f"✅ Saved: {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--style", type=str, default="kodak", help="LoRA style")
    parser.add_argument("--strength", type=float, default=0.1, help="Strength of img2img")
    parser.add_argument("--guidance", type=float, default=5.0, help="Guidance scale")
    args = parser.parse_args()

    input_dir = "./input"
    for file in os.listdir(input_dir):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            apply_lora(
                os.path.join(input_dir, file),
                args.style,
            )
