import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import os
from matplotlib import pyplot as plt
import torch.optim as optim
import time
from vit_pytorch import ViT


class VanillaViT(nn.Module):
    def __init__(self, image_size=72, patch_size=6, num_classes=1, dim=512, depth=6, heads=12,
                 mlp_dim=2048, dropout=0.1, emb_dropout=0.1):
        super(VanillaViT, self).__init__()
        self.model = ViT(image_size=image_size, patch_size=patch_size, num_classes=num_classes, dim=dim, depth=depth,
                         heads=heads, mlp_dim=mlp_dim, dropout=dropout, emb_dropout=emb_dropout)

    def forward(self, x):
        x = self.model(x)
        output = torch.sigmoid(x)

        return output


class Towers(Dataset):
    def __init__(self, csv_file, root_dir, transforms=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transforms

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path).convert('RGB')
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return image, y_label


IMG_SIZE = 72
transformations = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


BATCH_SIZE = 32

train_data = Towers(
    csv_file='train_revised.csv',
    root_dir='/home/admin1/CSD/ALaM/Jom/Vision_Transformer/train_combined/',
    transforms=transformations)

valid_data = Towers(
    csv_file='valid.csv',
    root_dir='/home/admin1/CSD/ALaM/Jom/Vision_Transformer/valid/',
    transforms=transformations)

train_loader = DataLoader(
    dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(
    dataset=valid_data, batch_size=BATCH_SIZE, shuffle=True)


vit1 = VanillaViT()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
vit1.to(device)

criterion = nn.BCELoss()
optimizer = optim.AdamW(vit1.parameters(), lr=0.01)

print('Training vanilla ViT...')
start = time.time()
for epoch in range(50):

    train_acc = 0.0
    train_loss = 0.0
    vit1.train()
    for data in train_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = vit1(inputs)
        labels = labels.unsqueeze(1).float()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_acc += (outputs.detach().round() == labels).float().mean()
        train_loss += float(loss)

    val_acc = 0.0
    val_loss = 0.0
    vit1.eval()
    for data in valid_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = vit1(inputs)
        labels = labels.unsqueeze(1).float()
        loss = criterion(outputs, labels)

        val_acc += (outputs.detach().round() == labels).float().mean()
        val_loss += float(loss)

    print(f'Epoch {epoch+1} - train_acc {train_acc / len(train_loader)}, train_loss {train_loss / len(train_loader)}, val_acc: {val_acc / len(train_loader)}, val_loss: {val_loss / len(train_loader)}')

print(f'Training took {(time.time() - start)/60} minutes \n.')
