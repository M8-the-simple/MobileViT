from utils import get_embedding
import numpy as np


emb1 = get_embedding(r"C:\Users\matia\Documents\RiTeh\6_semestar\Zavrsni_rad\HaarCascade\MobileViT\dataset\train\Mathias\00001.png")
emb2 = get_embedding(r"C:\Users\matia\Documents\RiTeh\6_semestar\Zavrsni_rad\HaarCascade\MobileViT\dataset\train\Mathias\00010.png")

print("Sličnost između ove dvije značajke: ", np.dot(emb1, emb2))
