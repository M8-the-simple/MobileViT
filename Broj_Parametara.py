import timm
import torch
import torch.nn.functional as F

model = timm.create_model("hf_hub:gaunernst/convnext_atto.cosface_ms1mv3", pretrained=True).eval()
embs = model(torch.randn(1, 3, 112, 112))  # output shape (1, 512)
embs = F.normalize(embs, dim=1)  # model output is not normalized

num_params = sum(p.numel() for p in model.parameters())

print(num_params)
