import timm
import torch
from models import get_embedding_model 

def get_timm_model(numclasses=0, name: str = "mobilevitv2_050.cvnets_in1k"):
    
    model = timm.create_model(name, pretrained=True, num_classes=numclasses)
    #model = timm.create_model("convnext_tiny", pretrained=True, num_classes=numclasses)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    name = name.replace(":", "_")
    name = name.replace("/", "_")
    model.name = name
    model.to(device)
    return model, device

model, device = get_embedding_model(numclasses=0)

data_config = timm.data.resolve_data_config(model.pretrained_cfg)

def get_timm_transform():
    transform = timm.data.create_transform(**data_config, is_training=False)
    print(f"{model.name}")
    if(model.name[0:7] == "hf_hub"):
        transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    return transform