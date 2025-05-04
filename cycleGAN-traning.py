import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from resNet_generator_model import ResNetGenerator
from patchGAN_discriminator import PatchDiscriminator
from datapreparation import ImageDataset, transformer
import os
from torchvision.utils import save_image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

real_dir = "dataset/trainA"
ghibli_dir = "dataset/trainB_ghibli"
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

num_epochs = 100
batch_size = 1
lr = 0.0002
image_size = 256

dataset = ImageDataset(root_dir="./dataset", train_a="trainA", train_b="trainB_ghibli", transformer=transformer)

dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0
)

G = ResNetGenerator(input_nc=3, output_nc=3).to(device) # real -> ghibli
F = ResNetGenerator(input_nc=3, output_nc=3).to(device) # ghibli -> real
D_A = PatchDiscriminator(3).to(device) # real
D_B = PatchDiscriminator(3).to(device) # ghibli

mse = nn.MSELoss()
l1 = nn.L1Loss()

g_optimizer = torch.optim.Adam(list(G.parameters()) + list(F.parameters()), lr=lr, betas=(0.5, 0.999))
d_A_optimizer = torch.optim.Adam(D_A.parameters(), lr=lr, betas=(0.5, 0.999))
d_B_optimizer = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(0.5, 0.999))

real_label = lambda x: torch.ones_like(x, device=device)
fake_label = lambda x: torch.zeros_like(x, device=device)

print("Starting training.....")

for epoch in range(1):
    for i, batch in enumerate(dataloader):
        real_A, real_B = batch["A"].to(device), batch["B"].to(device)

        # generator forward pass
        fake_B = G(real_A)
        rec_A = F(fake_B)

        fake_A = F(real_B)
        rec_b = G(fake_A)

        # adversarial loss
        loss_G_adv = mse(D_B(fake_B), real_label(D_B(fake_B)))
        loss_F_adv = mse(D_A(fake_A), real_label(D_A(fake_A)))

        # cycle consistency loss
        loss_cycle = l1(rec_A, real_A)+l1(rec_b, real_B)

        # identity loss
        loss_identity = l1(G(real_B), real_B) + l1(F(real_A), real_A)

        # total generator loss
        total_g_loss = loss_G_adv+loss_F_adv+10*loss_cycle + 5* loss_identity

        g_optimizer.zero_grad()
        total_g_loss.backward()
        g_optimizer.step()

        # discriminator A
        loss_D_A = (
            mse(D_A(real_A), real_label(D_A(real_A))) +
            mse(D_A(fake_A.detach()), fake_label(D_A(fake_A.detach())))
        ) * 0.5
        d_A_optimizer.zero_grad()
        loss_D_A.backward()
        d_A_optimizer.step()

        # === Discriminator B ===
        loss_D_B = (
            mse(D_B(real_B), real_label(D_B(real_B))) +
            mse(D_B(fake_B.detach()), fake_label(D_B(fake_B.detach())))
        ) * 0.5
        d_B_optimizer.zero_grad()
        loss_D_B.backward()
        d_B_optimizer.step()

        if (i + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Step [{i+1}/{len(dataloader)}] "
                  f"G Loss: {total_g_loss.item():.4f} D_A Loss: {loss_D_A.item():.4f} D_B Loss: {loss_D_B.item():.4f}")

    # Save generated sample
    if (epoch + 1) % 10 == 0 or True:
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            'G_state_dict': G.state_dict(),
            'F_state_dict': F.state_dict(),
            'D_A_state_dict': D_A.state_dict(),
            'D_B_state_dict': D_B.state_dict(),
            'g_optimizer': g_optimizer.state_dict(),
            'd_A_optimizer': d_A_optimizer.state_dict(),
            'd_B_optimizer': d_B_optimizer.state_dict(),
            'epoch': epoch
        }, f"checkpoints/cyclegan_epoch_{epoch+1}.pth")
        print(f"✅ Model saved at epoch {epoch+1}")
        save_image(fake_B * 0.5 + 0.5, f"{output_dir}/epoch_{epoch+1}_fakeB.png")
        save_image(fake_A * 0.5 + 0.5, f"{output_dir}/epoch_{epoch+1}_fakeA.png")


print("training complete")

