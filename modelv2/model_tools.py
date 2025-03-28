import torch
import torch.nn as nn

def save_model_weights(model: nn.Module, path: str) -> None:
    """
    Modelin ağırlıklarını belirtilen dosyaya kaydeder.
    """
    print(path)
    torch.save(model.state_dict(), path)
    print(f"Model weights saved to {path}")

def load_model(path: str, model: nn.Module) -> None:
    """
    Modelin ağırlıklarını belirtilen dosyadan yükler.
    """
    model.load_state_dict(torch.load(path, map_location=torch.device('cuda')))
    model.eval()  # Evaluation moduna al
    print("Model weights loaded successfully!")

def update_model_for_new_classes(model: nn.Module, new_num_classes: int) -> None:
    """
    Modelin çıkış katmanını yeni sınıf sayısına göre günceller.
    """
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, new_num_classes)
    print(f"Model output layer updated for {new_num_classes} classes.")

def remove_unmatched_layers(state_dict: dict, model: nn.Module) -> dict:
    """
    Yeni model ile uyumsuz olan katmanları kaldırarak yüklenebilir state_dict döndürür.
    """
    model_state = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
    return filtered_state_dict

def load_model_with_flexible_weights(path: str, model: nn.Module) -> None:
    """
    Modelin ağırlıklarını yükler ve eski-yeni model farklarını otomatik olarak yönetir.
    """
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    filtered_state_dict = remove_unmatched_layers(state_dict, model)
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    print("Model weights loaded with flexible adaptation!")


def save_model_full(path: str, model: nn.Module) -> None:
    """
    Modelin ağırlıklarını belirtilen dosyaya kaydeder.
    önerilmez
    """
    torch.save(model, path)
    print(f"Model weights saved to {path}")