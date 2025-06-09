import torch
import torch.nn as nn
from torch.optim import Adam
import timm
from sklearn.metrics import accuracy_score
from data_loader import PlantDataset
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_model(train_loader, val_loader, model, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                preds.extend(torch.argmax(out, dim=1).cpu().numpy())
                truths.extend(lbls.cpu().numpy())
        acc = accuracy_score(truths, preds)
        print(f"Epoch {epoch + 1}/{epochs} → Train Loss: {epoch_loss:.4f}  Val Accuracy: {acc:.4f}")

if __name__ == "__main__":
    # Model setup
    model = timm.create_model('resnet50', pretrained=True, num_classes=38)
    model = model.to(device)

    # Optimizer & loss
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Dataset & DataLoader
    train_ds = PlantDataset('path_to_train_data')
    val_ds = PlantDataset('path_to_val_data')
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    train_model(train_loader, val_loader, model, criterion, optimizer, epochs=5)
