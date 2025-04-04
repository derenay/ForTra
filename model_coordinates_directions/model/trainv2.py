import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import json
from dataset_indexed import FormationDataset, collate_fn
from model import HierarchicalFormationTransformer
from model_tools import save_model_weights, load_model
from data_split import split_ata


# Modelin weights kaydedilecek yolu
path = "trained_models/saved_e.pth"

# JSON dosyasından veri setini yükle
with open('dataset/data.json', 'r') as f:
    data_samples = json.load(f)

# Tüm dataset oluşturuluyor
full_dataset = FormationDataset(data_samples)

train_dataset,test_dataset = split_ata(full_dataset)


# DataLoader'lar
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                          collate_fn=collate_fn, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                         collate_fn=collate_fn, num_workers=4, pin_memory=True)

# Modeli oluştur
model = HierarchicalFormationTransformer(
    coord_dim=2,
    class_vocab_size=10,
    class_embed_dim=16,
    stage_dims=[512, 384, 256],
    num_heads=16,
    num_layers=[16, 16, 16],
    num_formations=8,
    dropout_stages=[0.5, 0.3, 0.2],
    use_adapter=True,
    adapter_dim=64,
    pos_type='learnable'
)

# Modeli yüklemek isterseniz, aşağıdaki satırı aktif edebilirsiniz.
# load_model(path, model)

# Cihaz ayarı: GPU varsa GPU, yoksa CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Optimizer, loss fonksiyonu ve LR scheduler
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()  # Hedefler integer indeksler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Değerlendirme fonksiyonu: hem loss hem de accuracy hesaplar
def evaluate(loader):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch_coords, batch_classes, batch_labels in loader:
            batch_coords = batch_coords.to(device)
            batch_classes = batch_classes.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_coords, batch_classes)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item() * batch_labels.size(0)
            
            _, preds = torch.max(outputs, dim=1)
            total_correct += (preds == batch_labels).sum().item()
            total_samples += batch_labels.size(0)
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

num_epochs = 20
model.train()

for epoch in range(num_epochs):
    print("entered")
    total_loss = 0.0
    total_train_correct = 0
    total_train_samples = 0
    
    for batch_coords, batch_classes, batch_labels in train_loader:
        batch_coords = batch_coords.to(device)
        batch_classes = batch_classes.to(device)
        batch_labels = batch_labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_coords, batch_classes)  
        loss = criterion(outputs, batch_labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * batch_labels.size(0)
        
        # Hesaplama için doğru tahminler
        _, preds = torch.max(outputs, dim=1)
        total_train_correct += (preds == batch_labels).sum().item()
        total_train_samples += batch_labels.size(0)
    
    # Epoch sonu ortalama eğitim loss ve accuracy
    train_loss = total_loss / total_train_samples
    train_accuracy = total_train_correct / total_train_samples
    
    # Test metriklerini hesapla
    test_loss, test_accuracy = evaluate(test_loader)
    
    # Learning rate scheduler adım atar
    scheduler.step()
    
    # Model ağırlıklarını kaydet
    save_model_weights(model=model, path=path)
    
    print(f"Epoch {epoch+1}/{num_epochs}:")
    print(f"    Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.3f}")
    print(f"    Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.3f}")

# Son model kaydediliyor
save_model_weights(model=model, path=path)
print("Training complete and model saved!")
