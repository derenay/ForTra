import torch
import argparse
import json
from model import HierarchicalFormationTransformer
from model_tools import load_model
import pandas as pd
import numpy as np


def predict_formation(model, coords, class_tokens):
    coords = torch.tensor(coords, dtype=torch.float32).unsqueeze(0)  # (1, N, 2)
    class_tokens = torch.tensor(class_tokens, dtype=torch.long).unsqueeze(0)  # (1, N)
    
    with torch.no_grad():
        output = model(coords, class_tokens)
        prediction = torch.argmax(output, dim=-1).item()
    
    return prediction


formation2idx = {
            "Line": 0,
            "Wedge": 1,
            "Vee": 2,
            "Echelon Right": 3,
            "Herringbone": 4,
            "Coil": 5,
            "Staggered Column": 6,
            "Echelon Left": 7,
            "Column": 8,
          
        }


def main():
    # Modeli oluştur ve yükle
    model = HierarchicalFormationTransformer(
      coord_dim=2,
      class_vocab_size=10,
      class_embed_dim=16,
      stage_dims=[512, 384, 256],
      num_heads=16,
      num_layers=[16, 16, 16],
      num_formations=9,
      dropout_stages=[0.5, 0.3, 0.2],
      use_adapter=True,
      adapter_dim=64,
      pos_type='learnable'
  )

    load_model("/home/earsal@ETE.local/Desktop/codes/military transformation/saved_e.pth", model)
    
    # JSON formatında koordinat ve sınıf verilerini oku
    
    
    df = pd.read_json("dataset/val.json")
    df['classes'] = df['classes'].apply(lambda x : [0 if item=="tank" else item for item in x])
    df['formation'] = df['formation'].apply(lambda x: formation2idx.get(x, -1)).astype(int)

   
    predicted = []
    original = []
    for coordinates,classes,formation in zip(df['coordinates'], df['classes'], df['formation']):
        a = predict_formation(model, coordinates, classes)
        original.append(formation)
        predicted.append(a) 
   
    predicted = np.array(predicted)
    original = np.array(original)
    
    lenghtof = len(predicted)
    
    count = np.sum(predicted == original)  # Çok daha hızlı!
        
    print(f"lenghtof {lenghtof}   count{count}")
    print((count/lenghtof)*100)
    print(predicted, original)
    
if __name__ == "__main__":
   
    main()
