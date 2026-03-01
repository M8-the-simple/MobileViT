import timm
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from timm.loss import LabelSmoothingCrossEntropy
from tqdm import tqdm

def imagenet_validate(model, val_loader, device):
    
    model.eval()
    criterion = LabelSmoothingCrossEntropy()
    
    top1, top5 = 0., 0.
    total = 0.
    
    with torch.no_grad():
        for images, target in tqdm(val_loader):
            images = images.to(device)
            target = target.to(device)
            
            output = model(images)
            loss = criterion(output, target)
            
            # Top-1 i Top-5 accuracy
            _, pred = output.topk(5, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            
            top1 += correct[0].sum().item()
            top5 += correct[:5].sum().item()
            total += target.size(0)
    
    top1_acc = 100. * top1 / total
    top5_acc = 100. * top5 / total
    return top1_acc, top5_acc
def main():

    # Korištenje
    model_name = "mobilevitv2_050.cvnets_in1k"
    model = timm.create_model(model_name, pretrained=True, num_classes=1000)

    # Transform + dataset
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config)
    dataset = datasets.ImageFolder("C:\\Dev\\Izborni_Projekt\\imagenet1k", transform=transform)
    loader = DataLoader(dataset, batch_size=16, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    top1, top5 = imagenet_validate(model, loader, device)
    print(f"Top-1: {top1:.2f}%, Top-5: {top5:.2f}%")

if __name__ == "__main__":
    main()
