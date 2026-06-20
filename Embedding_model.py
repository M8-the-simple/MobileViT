
import timm
import torch


def get_embedding_model(numclasses=0):
    model = timm.create_model("mobilevitv2_050.cvnets_in1k", pretrained=True, num_classes=numclasses)
    #model = timm.create_model("convnext_tiny", pretrained=True, num_classes=numclasses)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, device

model, device = get_embedding_model(numclasses=0)

data_config = timm.data.resolve_data_config(model.pretrained_cfg)
transform = timm.data.create_transform(**data_config, is_training=False)


def get_transform():
    return transform