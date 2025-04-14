import torch
from torch.utils.data import Dataset

class FormationDataset(Dataset):
    def __init__(self, data):
        """
        data: Her biri aşağıdaki yapıya sahip örneklerden oluşan bir liste:
              {
                  'coordinates': [[x, y], [x, y], ...],
                  'classes': ['tank', 'tank', 'ifv', ...],
                  'formation': 'Vee'  # veya diğer formasyonlar
              }
        """
        self.data = data
        # Sınıf bilgilerini embedding için index'e çeviriyoruz
        self.class2idx = {'tank': 0, 'ifv': 1}
        
        # Formasyon etiketleri; nihai sınıflandırma için kullanılacak
        self.formation2idx = {
             "Line": 0,
            "Wedge": 1,
            "Vee": 2,
            "Herringbone": 3,
            "Coil": 4,
            "Staggered Column": 5,
            "Column": 6,
            "Echelon": 7 
        }

        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        # Koordinatlar: [[x, y], ...] → float tensora çevriliyor
        coords = torch.tensor(sample['coordinates'], dtype=torch.float)
        # Sınıf bilgileri: Her nesne için string değer, embedding için index'e dönüştürülüyor
        classes = torch.tensor(sample['classes'], dtype=torch.long)
        # Formasyon etiketi: Sınıflandırma hedefi
        formation_label = torch.tensor(sample['formation'], dtype=torch.long)
        return coords, classes, formation_label

def collate_fn(batch):
    """
    Batch içerisindeki örnekler farklı uzunlukta sequence'lere sahip olabilir.
    Bu fonksiyon, tüm örnekleri aynı uzunlukta olacak şekilde padleyip batch oluşturur.
    """
    max_len = max(sample[0].shape[0] for sample in batch)
    batch_coords, batch_classes, batch_labels = [], [], []
    for coords, classes, label in batch:
        pad_len = max_len - coords.shape[0]
        if pad_len > 0:
            # Koordinatlar için pad: 0 ile doldurulur.
            pad_coords = torch.zeros(pad_len, coords.shape[1])
            coords = torch.cat([coords, pad_coords], dim=0)
            # Sınıf bilgileri için pad: 0 ile doldurulur.
            pad_classes = torch.zeros(pad_len, dtype=torch.long)
            classes = torch.cat([classes, pad_classes], dim=0)
        batch_coords.append(coords.unsqueeze(0))
        batch_classes.append(classes.unsqueeze(0))
        batch_labels.append(label.unsqueeze(0))
    
    batch_coords = torch.cat(batch_coords, dim=0)
    batch_classes = torch.cat(batch_classes, dim=0)
    batch_labels = torch.cat(batch_labels, dim=0)
    
    return batch_coords, batch_classes, batch_labels
