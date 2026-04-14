import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

# --- Část A: Klasická (plně propojená) neuronová síť (MLP) ---
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        # Vstup: 3 kanály * 32 * 32 = 3072
        self.network = nn.Sequential(
            nn.Linear(3072, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 43)  # 43 tříd dopravních značek
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.network(x)
        return logits

# --- Část B: Konvoluční neuronová síť (CNN) ---
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Konvoluční část
        self.features = nn.Sequential(
            # Blok 1: 32x32 -> MaxPool -> 16x16
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            # Blok 2: 16x16 -> MaxPool -> 8x8
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Klasifikační část
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # 32 kanálů * 8 * 8 pixelů = 2048
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 43)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_and_evaluate(model, train_loader, test_loader, epochs=5):
    model.to(device)
    # Ztrátová funkce (pro klasifikaci do více tříd)
    criterion = nn.CrossEntropyLoss()
    # Optimalizátor (Adam je často rychlejší než SGD)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"\n--- Zahájení trénování modelu: {model.__class__.__name__} ---")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()           # Vynulovat gradienty
            outputs = model(images)         # Dopředný průchod
            loss = criterion(outputs, labels) # Výpočet chyby
            loss.backward()                 # Zpětné šíření (backpropagation)
            optimizer.step()                # Aktualizace vah

            running_loss += loss.item()

        print(f"Epocha {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f}")

    total_time = time.time() - start_time
    print(f"Trénování dokončeno za {total_time:.2f} sekund.")

    # Evaluace (testování)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Přesnost na testovacích datech: {accuracy:.2f} %")
    return accuracy

# Definujeme transformace: 
# 1. Změna velikosti na 32x32 pixelů
# 2. Převod na PyTorch Tensor (změní pixely z 0-255 na 0.0-1.0)
data_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Volitelná normalizace
])

# Načtení trénovacích dat
train_data = datasets.GTSRB(
    root='./data', 
    split='train', 
    download=True, 
    transform=data_transforms
)

# Načtení testovacích dat
test_data = datasets.GTSRB(
    root='./data', 
    split='test', 
    download=True, 
    transform=data_transforms
)

print(f"Počet trénovacích obrázků: {len(train_data)}")
print(f"Počet testovacích obrázků: {len(test_data)}")

# Vytvoření DataLoaderů
BATCH_SIZE = 32 

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)

print("DataLoadery jsou připraveny.")
print(f"Počet dávek v train_loaderu: {len(train_loader)}")

# Vytvoření instancí modelů
mlp_model = MLP()
cnn_model = CNN()

print("Modely MLP a CNN byly úspěšně vytvořeny.")

# SPRAŽENÍ TRÉNINKU
# Nastavení zařízení (GPU pokud je k dispozici, jinak CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Trénování proběhne na: {device}")
# (Pro začátek zkusíme jen 3 epochy, abys viděl, jestli to funguje)
mlp_acc = train_and_evaluate(mlp_model, train_loader, test_loader, epochs=3)
cnn_acc = train_and_evaluate(cnn_model, train_loader, test_loader, epochs=3)