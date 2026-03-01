import timm 
import torch
import os
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

model = timm.create_model("mobilevitv2_050.cvnets_in1k", pretrained=True, num_classes=0)

model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

data_config = timm.data.resolve_data_config(model.pretrained_cfg)
transform = timm.data.create_transform(**data_config)

def get_embedding(image_path):
    img = Image.open(image_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(x)
    return emb.cpu().numpy().flatten()

# 10 poznatih ImageNet slika (možeš ih downloadati ili skinuti s neta)
imagenet_embs = []
imagenet_labels = []    

for class_name in os.listdir("imagenet1k")[:10]:
    class_path = os.path.join("imagenet1k", class_name)
    if not os.path.isdir(class_path):
        continue
    for img_name in os.listdir(class_path)[:10]: 
        img_path = os.path.join(class_path, img_name)
        emb = get_embedding(img_path)
        imagenet_embs.append(emb)
        imagenet_labels.append(class_name)

imagenet_embs = np.array(imagenet_embs)

imagenet_mean = np.mean(imagenet_embs, axis=1)

# t-SNE redukcija na 2D
pca = PCA(n_components=2, random_state=42)
emb_2d = pca.fit_transform(imagenet_mean)

plt.figure(figsize=(12, 5))

# ImageNet embeddingi (crvene)
plt.scatter(emb_2d[:len(imagenet_labels), 0], emb_2d[:len(imagenet_labels), 1], 
           c='red', label='ImageNet', s=100, alpha=0.8)

plt.legend()
plt.title('Embedding Space: ImageNet vs Face Recognition')
plt.xlabel('t-SNE 1')
plt.tight_layout()
plt.show()



