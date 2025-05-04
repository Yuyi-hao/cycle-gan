import torch
from resNet_generator_model import ResNetGenerator
from torchvision import transforms
from PIL import Image
import os

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
G = ResNetGenerator(input_nc=3, output_nc=3).to(device)
checkpoint = torch.load("./checkpoints/cyclegan_epoch_1.pth", map_location=device)
G.load_state_dict(checkpoint["G_state_dict"])
G.eval()


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load image
img_path = "./dataset/trainA/world_0045.jpg"
img = Image.open(img_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    fake_img = G(img_tensor)

# De-normalize and convert back to PIL
fake_img = fake_img.squeeze(0).cpu() * 0.5 + 0.5  # from [-1,1] to [0,1]
fake_pil = transforms.ToPILImage()(fake_img.clamp(0, 1))
fake_pil.save("ghibli_output.jpg")
print("🎉 Image saved as 'ghibli_output.jpg'")


