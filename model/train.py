import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from dataset import FormationDataset, collate_fn
from model import HierarchicalFormationTransformer
from model_tools import save_model_weights, load_model

path = "trained_models/hierarchical_formation_transformer_final.pth"


# JSON dosyasından veri setini yükle
with open('dataset/data.json', 'r') as f:
    data_samples = json.load(f)

# Dataset ve DataLoader oluşturuluyor
dataset = FormationDataset(data_samples)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True,
                         collate_fn=collate_fn, num_workers=4, pin_memory=True)

# Modeli oluştur
model = HierarchicalFormationTransformer(
    coord_dim=2,
    class_vocab_size=10,
    class_embed_dim=16,
    stage_dims=[512, 384, 256],
    num_heads=8,
    num_layers=[4, 4, 4],
    num_formations=10,
    dropout_stages=[0.5, 0.3, 0.2],
    use_adapter=True,
    adapter_dim=64,
    pos_type='learnable'
)

load_model(path, model)

# Cihaz ayarı: GPU varsa GPU, yoksa CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Optimizer, loss fonksiyonu ve LR scheduler
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()  # Hedefler [batch] şeklinde, integer indeksler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

num_epochs = 20
model.train()
for epoch in range(num_epochs):
    total_loss = 0.0
    for batch_coords, batch_classes, batch_labels in data_loader:
        batch_coords = batch_coords.to(device)
        batch_classes = batch_classes.to(device)
        batch_labels = batch_labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_coords, batch_classes)  
        loss = criterion(outputs, batch_labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    
    scheduler.step()
    avg_loss = total_loss / len(data_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")


save_model_weights(model, path)


print("Training complete and model saved!")




# Epoch 1/20, Loss: 2.3522
# Epoch 2/20, Loss: 2.3233
# Epoch 3/20, Loss: 2.1639
# Epoch 4/20, Loss: 1.8533
# Epoch 5/20, Loss: 1.6689
# Epoch 6/20, Loss: 1.1522
# Epoch 7/20, Loss: 0.9566
# Epoch 8/20, Loss: 0.8734
# Epoch 9/20, Loss: 0.7736
# Epoch 10/20, Loss: 0.6967
# Epoch 11/20, Loss: 0.6286
# Epoch 12/20, Loss: 0.5906
# Epoch 13/20, Loss: 0.5972
# Epoch 14/20, Loss: 0.5523
# Epoch 15/20, Loss: 0.5568
# Epoch 16/20, Loss: 0.5089
# Epoch 17/20, Loss: 0.4808
# Epoch 18/20, Loss: 0.4720
# Epoch 19/20, Loss: 0.4674
# Epoch 20/20, Loss: 0.4466
