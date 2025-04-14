import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
# from torch.utils.tensorboard import SummaryWriter # <-- Kaldırıldı
import json
import os
# import argparse # <-- Kaldırıldı
import time # For timestamping logs
import random
import numpy as np
from sklearn.model_selection import train_test_split # For splitting data

# --- Import your custom modules ---
try:
    from dataset import FormationDataset, collate_fn
    # <<< MODEL ADINI DOĞRU GİRDİĞİNİZDEN EMİN OLUN (model.py veya modeldenemesi.py) >>>
    from model import HierarchicalFormationTransformer # model.py kullandığınızı varsayıyorum
    from model_tools import save_model_weights, load_model # Assuming these exist
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    print("Please ensure dataset.py, model.py (or modeldenemesi.py), and model_tools.py are accessible.")
    exit()


def set_seed(seed:int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_accuracy(outputs, labels):
    """Calculates batch accuracy."""
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct / total if total > 0 else 0.0

# --- Main Training Function (config sözlüğünü alıyor) ---
def main(config:dict) -> None:
   
    set_seed(config['seed'])
    # run_name'i config içinden alıyoruz
    run_dir = os.path.join(config['save_dir'], config['run_name'])
    # log_dir artık TensorBoard için kullanılmıyor ama run_dir hala model kaydetmek için gerekli
    model_save_path = os.path.join(run_dir, 'best_model.pth')

    os.makedirs(run_dir, exist_ok=True)
    # writer = SummaryWriter(log_dir=log_dir) # <-- Kaldırıldı

    print(f"Starting Run: {config['run_name']}")
    print(f"Configuration:\n{config}")
    # print(f"Logs will be saved to: {log_dir}") # <-- Log dizini bilgisi kaldırıldı
    print(f"Best model will be saved to: {model_save_path}")

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading and Splitting ---
    print(f"Loading data from {config['data_file']}...")
    try:
        with open(config['data_file'], 'r') as f:
            data_samples = json.load(f)
    except FileNotFoundError:
        print(f"Error: Data file not found at {config['data_file']}")
        exit()
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {config['data_file']}")
        exit()

    print("Splitting data into training and validation sets...")
    if not data_samples:
        print("Error: No data samples loaded.")
        exit()

    try:
        train_idx, val_idx = train_test_split(
            list(range(len(data_samples))),
            test_size=config['validation_split'],
            random_state=config['seed']
        )
        if not train_idx or not val_idx:
             raise ValueError("Split resulted in empty training or validation set.")
        print(f"Training samples: {len(train_idx)}, Validation samples: {len(val_idx)}")
    except ValueError as e:
         print(f"Error during data splitting: {e}")
         print("Ensure dataset is large enough and validation_split is appropriate.")
         exit()

    full_dataset = FormationDataset(data_samples)
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    actual_class_vocab_size = len(full_dataset.class2idx)
    actual_num_formations = len(full_dataset.formation2idx)
    print(f"Detected Class Vocab Size: {actual_class_vocab_size}")
    print(f"Detected Number of Formations: {actual_num_formations}")

    print("Creating DataLoaders...")
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                              collate_fn=collate_fn, num_workers=config['num_workers'],
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False,
                            collate_fn=collate_fn, num_workers=config['num_workers'],
                            pin_memory=True)

    # --- Model Definition ---
    print("Initializing model...")
    model = HierarchicalFormationTransformer(
        num_formations=actual_num_formations,
        class_vocab_size=actual_class_vocab_size,
        class_embed_dim=config['class_embed_dim'],
        direction_dim=config['direction_dim'],
        coord_dim=2,
        stage_dims=config['stage_dims'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dropout_stages=config['dropout_stages'],
        use_adapter=config['use_adapter'],
        adapter_dim=config['adapter_dim'],
        pos_type=config['pos_type'],
        max_len=config['max_len'],
        ffn_ratio=config['ffn_ratio']
    ).to(device)

    # Optional: Load pre-trained weights
    if config['load_path']:
        print(f"Loading model weights from: {config['load_path']}")
        try:
            load_model(path=config['load_path'], model=model)
        except Exception as e:
            print(f"Warning: Failed to load model weights from {config['load_path']}. Error: {e}. Training from scratch.")


    # --- Optimizer, Loss, Scheduler ---
    print("Setting up optimizer, loss function, and scheduler...")
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_step_size'], gamma=config['lr_gamma'])

    # --- Training Loop ---
    print(f"Starting training for {config['epochs']} epochs...")
    best_val_loss = float('inf')

    for epoch in range(config['epochs']):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0.0
        total_train_acc = 0.0
        num_train_batches = 0

        # Basit ilerleme göstergesi (isteğe bağlı)
        print(f"\n--- Epoch {epoch+1}/{config['epochs']} ---")
        start_time_epoch = time.time()

        for batch_idx, (coords, classes, directions, padding_mask, labels) in enumerate(train_loader):
            coords, classes, directions, padding_mask, labels = \
                coords.to(device), classes.to(device), directions.to(device), padding_mask.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(coords, classes, directions, key_padding_mask=padding_mask)
            loss = criterion(outputs, labels)
            loss.backward()

            if config['grad_clip_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['grad_clip_norm'])

            optimizer.step()

            total_train_loss += loss.item()
            total_train_acc += calculate_accuracy(outputs, labels)
            num_train_batches += 1

            # İlerleme göstergesi (her N batch'te bir)
            if (batch_idx + 1) % 10 == 0: # Her 10 batch'te bir yazdır
                 print(f"  Batch {batch_idx+1}/{len(train_loader)} | Train Loss (batch): {loss.item():.4f}", end='\r')


        avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0.0
        avg_train_acc = total_train_acc / num_train_batches if num_train_batches > 0 else 0.0

        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0.0
        total_val_acc = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for coords, classes, directions, padding_mask, labels in val_loader:
                coords, classes, directions, padding_mask, labels = \
                    coords.to(device), classes.to(device), directions.to(device), padding_mask.to(device), labels.to(device)

                outputs = model(coords, classes, directions, key_padding_mask=padding_mask)
                loss = criterion(outputs, labels)

                total_val_loss += loss.item()
                total_val_acc += calculate_accuracy(outputs, labels)
                num_val_batches += 1

        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
        avg_val_acc = total_val_acc / num_val_batches if num_val_batches > 0 else 0.0

        # --- Logging (Console) ---
        epoch_duration = time.time() - start_time_epoch
        current_lr = scheduler.get_last_lr()[0]
        # Önceki ilerleme satırını temizle ve sonuçları yazdır
        print(" " * 80, end='\r') # Önceki satırı temizle
        print(f"Epoch {epoch+1}/{config['epochs']} | Time: {epoch_duration:.2f}s | "
              f"LR: {current_lr:.6f} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}")

        # Log metrics to TensorBoard <-- Kaldırıldı
        # writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        # ... diğer writer çağrıları kaldırıldı ...

        # --- Scheduler Step ---
        scheduler.step()

        # --- Save Best Model ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_model_weights(model=model, path=model_save_path)
            print(f"  -> New best model saved with Val Loss: {best_val_loss:.4f}")

    # --- End of Training ---
    # writer.close() # <-- Kaldırıldı
    print("-" * 20)
    print("Training complete!")
    print(f"Best validation loss achieved: {best_val_loss:.4f}")
    print(f"Best model weights saved to: {model_save_path}")
    # TensorBoard log bilgisi kaldırıldı
    print("-" * 20)



if __name__ == "__main__":
    # --- Configuration  ---
    config = {
        # Data and Save Paths
        'data_file': 'el.json',
        'save_dir': 'trained_models',
        'run_name': f'hft_balanced_run_{time.strftime("%Y%m%d_%H%M%S")}', 

        # Model Hyperparameters
        'class_embed_dim': 32,      
        'direction_dim': 1,            
        'stage_dims': [256, 128],      
        'num_heads': 8,                
        'num_layers': [6, 6],         
        'dropout_stages': [0.2,0.1], 
        'use_adapter': True,          
        'adapter_dim': 32,             
        'pos_type': 'learnable',
        'max_len': 50,               
        'ffn_ratio': 4,                


        # Training Hyperparameters
        'epochs': 20,                  
        'batch_size': 32,             
        'lr': 0.0001,                 
        'weight_decay': 0.01,          
        'lr_step_size': 15,          
        'lr_gamma': 0.5,            
        'grad_clip_norm': 1.0,
        'validation_split': 0.15,
        'num_workers': 4,
        'seed': 42,
        'load_path': 'trained_models/hft_balanced_run_20250414_090453/best_model.pth'
    }

    # Eğitim fonksiyonunu çağır
    main(config)