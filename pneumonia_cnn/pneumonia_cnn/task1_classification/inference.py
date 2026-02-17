import torch
import torch.nn.functional as F

def predict_image(model, image_tensor, device):
    model.eval()
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    return predicted.item(), confidence.item()


def get_predictions(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    misclassified = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for i in range(len(preds)):
                if preds[i] != labels[i]:
                    misclassified.append(
                        (images[i].cpu(), labels[i].item(), preds[i].item())
                    )

    return all_preds, all_labels, misclassified
