import os
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms, models
from torchvision.models import vgg19, VGG19_Weights

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载VGG模型
vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()


# 内容和风格层
content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# 图像预处理和反处理
def load_image(path, shape=None, max_size=512):
    image = Image.open(path).convert('RGB')
    if shape:
        size = tuple([int(s) for s in shape])
    else:
        size = max(image.size)
        if size > max_size:
            size = max_size
        size = (size, size)

    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[:3, :, :]),  # RGB
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    image = in_transform(image).unsqueeze(0)
    return image.to(device)

def im_convert(tensor):
    image = tensor.cpu().clone().detach()
    image = image.squeeze(0)
    image = transforms.Normalize(
        mean=[-2.12, -2.04, -1.80],
        std=[4.37, 4.46, 4.44])(image)
    image = torch.clamp(image, 0, 1)
    return transforms.ToPILImage()(image)

# 内容损失和风格损失计算模块（略简化）
def get_features(image, model, layers=None):
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if layers and f"conv_{name}" in layers:
            features[f"conv_{name}"] = x
    return features

def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

def style_transfer(content, style, steps=300, style_weight=1e6, content_weight=1):
    content_features = get_features(content, vgg, content_layers + style_layers)
    style_features = get_features(style, vgg, content_layers + style_layers)

    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    target = content.clone().requires_grad_(True).to(device)
    optimizer = torch.optim.Adam([target], lr=0.003)

    for i in range(steps):
        target_features = get_features(target, vgg, content_layers + style_layers)
        content_loss = torch.mean((target_features['conv_4'] - content_features['conv_4']) ** 2)

        style_loss = 0
        for layer in style_layers:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            style_gram = style_grams[layer]
            layer_loss = torch.mean((target_gram - style_gram) ** 2)
            style_loss += layer_loss / (target_feature.shape[1] ** 2)

        total_loss = content_weight * content_loss + style_weight * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    return target

# ----------------------------
# 批量处理风格迁移
# ----------------------------
def batch_style_transfer(input_dir, output_dir, style_path):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    style_image = load_image(style_path)

    for path in input_dir.glob("**/*"):
        if path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
        try:
            print(f"Processing {path.name}")
            content_image = load_image(path, shape=style_image.shape[-2:])
            output = style_transfer(content_image, style_image)
            save_path = output_dir / path.relative_to(input_dir)
            save_path = save_path.with_suffix(".jpg")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            im_convert(output).save(save_path)
        except Exception as e:
            print(f"Failed on {path.name}: {e}")

# ✅ 示例调用
if __name__ == "__main__":
    batch_style_transfer(
        input_dir="G:/3/reinhard/yuan",                    # 输入图像文件夹
        output_dir="G:/3/style_transferred",      # 输出文件夹
        style_path="F:/ranse/normalpicture.png"   # 风格参考图像
    )
