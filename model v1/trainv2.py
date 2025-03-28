from torch.utils.data import random_split

# JSON dosyasından veri setini yükle
with open('dataset/data.json', 'r') as f:
    data_samples = json.load(f)

# Dataset oluştur
dataset = FormationDataset(data_samples)

# Veri setini eğitim ve validasyon olarak böl
train_size = int(0.8 * len(dataset))  # %80 eğitim
val_size = len(dataset) - train_size  # %20 validasyon
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoader'ları oluştur
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                          collate_fn=collate_fn, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                        collate_fn=collate_fn, num_workers=4, pin_memory=True)




def validate_model(model, val_loader, criterion, device):
    model.eval()  # Modeli değerlendirme moduna geçirin
    total_loss = 0.0
    with torch.no_grad():  # Gradyan hesaplamalarını devre dışı bırakın
        for batch_coords, batch_classes, batch_labels in val_loader:
            batch_coords = batch_coords.to(device)
            batch_classes = batch_classes.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_coords, batch_classes)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss



num_epochs = 20
model.train()

for epoch in range(num_epochs):
    # Eğitim aşaması
    model.train()
    total_train_loss = 0.0
    for batch_coords, batch_classes, batch_labels in train_loader:
        batch_coords = batch_coords.to(device)
        batch_classes = batch_classes.to(device)
        batch_labels = batch_labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_coords, batch_classes)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(train_loader)
    
    # Validasyon aşaması
    model.eval()
    avg_val_loss = validate_model(model, val_loader, criterion, device)
    
    # Learning rate scheduler adımını uygula
    scheduler.step()
    
    # Modeli kaydet
    save_model_weights(model=model, path=path)
    
    # Sonuçları yazdır
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

print("Training complete and model saved!")


import matplotlib.pyplot as plt

# Loss değerlerini sakla
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # Eğitim ve validasyon işlemleri...
    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = validate_model(model, val_loader, criterion, device)
    
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

# Loss eğrilerini çiz
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.show()