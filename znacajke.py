import timm 
import torch
import os
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
import numpy as np
from Embedding_model import get_embedding_model
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

model, device = get_embedding_model()
data_config = timm.data.resolve_data_config(model.pretrained_cfg)
transform = timm.data.create_transform(**data_config, is_training=False)
NUM_CLASSES = 10
imagenet_classes = {
    'n01440764': 'Tench',           
    'n01443537': 'Goldfish',
    'n01484850': 'Great_white_shark',
    'n01491361': 'Tiger_shark', 
    'n01494475': 'Hammerhead',
    'n01496331': 'Electric_ray', 
    'n01498041': 'Stingray',
    'n01514668': 'Rooster',
    'n01514859': 'Hen',
    'n01518878': 'Ostrich'
}

def get_embedding(image_path):
    img = Image.open(image_path).convert('RGB')
    if img is None:
        raise ValueError(f"Ne mogu otvoriti sliku: {image_path}")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(x)
    emb_np = emb.cpu().numpy().flatten()
    return emb_np

all_embeddings = []
all_labels = []  
for class_id in os.listdir("imagenet-val")[:NUM_CLASSES]:
    
    class_path = os.path.join("imagenet-val", class_id)
    if not os.path.isdir(class_path):
        continue
    class_name = imagenet_classes[class_id]
    
    for img_name in os.listdir(class_path): 
        img_path = os.path.join(class_path, img_name)
        emb = get_embedding(img_path)
        all_embeddings.append(emb)
        all_labels.append(class_name)
    print(f"Obrađena klasa: {class_name}, zasad {len(all_embeddings)} slika.")
for class_name in os.listdir("dataset/train"):
    class_path = os.path.join("dataset/train", class_name)
    if not os.path.isdir(class_path):
        continue
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        emb = get_embedding(img_path)
        all_embeddings.append(emb)
        all_labels.append(class_name)
    print(f"Obrađena osoba: {class_name}, zasad {len(all_embeddings)} slika.")

all_embeddings = np.array(all_embeddings)

pca = PCA(n_components=2)
centroids_2d = pca.fit_transform(all_embeddings)

plt.figure(figsize=(10, 5))
#plt.subplot(1, 2, 1)
plt.scatter(centroids_2d[:NUM_CLASSES * 50, 0], 
           centroids_2d[:NUM_CLASSES * 50, 1],
           c='red', s=50, label='ImageNet-1k', alpha=0.8)
plt.scatter(centroids_2d[NUM_CLASSES * 50:, 0], 
           centroids_2d[NUM_CLASSES * 50:, 1],
           c='blue', s=50, label='Lica', alpha=0.8)
plt.legend()
plt.title('Analiza glavnih komponenti (PCA) vektora lica i imageNet-1k')


# for i, label in enumerate(all_labels):
#     plt.annotate(label[:10], (centroids_2d[i, 0], centroids_2d[i, 1]), 
#                 xytext=(5, 5), textcoords='offset points')
#plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='gray', s=150)
#plt.title('Centroidski vektori s oznakama')
#plt.tight_layout()
#plt.savefig('vektori_pca.png', dpi=300)
plt.show()



