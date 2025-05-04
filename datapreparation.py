import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

transformer = transforms.Compose([
    transforms.Resize(286),
    transforms.RandomCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])

class ImageDataset(Dataset):
    def __init__(self, root_dir, train_a, train_b, transformer=None):
        self.transformer = transformer

        self.dir_A = os.path.join(root_dir, train_a)
        self.dir_B = os.path.join(root_dir, train_b)

        self.files_A = os.listdir(self.dir_A)
        self.files_B = os.listdir(self.dir_B)

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
    
    def __getitem__(self, index):
        img_A_path = os.path.join(self.dir_A, self.files_A[index%len(self.files_A)])
        img_B_path = os.path.join(self.dir_B, self.files_B[index%len(self.files_B)])

        img_A = Image.open(img_A_path).convert('RGB')
        img_B = Image.open(img_B_path).convert('RGB')

        if self.transformer:
            img_A = self.transformer(img_A)
            img_B = self.transformer(img_B)
        
        return {"A": img_A, "B": img_B}


def show_img(img1, img2, title1="Real", title2="Ghibli"):
    fix, axis = plt.subplots(1, 2, figsize=(8, 4))
    axis[0].imshow(TF.to_pil_image((img1*0.5+0.5)))
    axis[0].set_title(title1)
    axis[0].axis("off")
    axis[1].imshow(TF.to_pil_image((img2*0.5+0.5)))
    axis[1].set_title(title2)
    axis[1].axis("off")
    
    plt.show()

dataset = ImageDataset(root_dir="./dataset", train_a="trainA", train_b="trainB_ghibli", transformer=transformer)

dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0
)

if __name__=="__main__":
    for i, batch in enumerate(dataloader):
        print(batch["A"].shape, batch["B"].shape)
        show_img(batch["A"].squeeze(0), batch["B"].squeeze(0))
        if i > 2:
            break

