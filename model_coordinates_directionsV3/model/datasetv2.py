import torch
from torch.utils.data import Dataset
# Assume torch.nn is imported if needed elsewhere, not directly needed here

class FormationDataset(Dataset):
    def __init__(self, data):
        """
        data: Her biri aşağıdaki yapıya sahip örneklerden oluşan bir liste:
              {
                  'coordinates': [[x, y], [x, y], ...],
                  'classes': ['tank', 'tank', 'ifv', ...],
                  'formation': 'Vee',  # veya diğer formasyonlar
                  'directions': [0.781, 0.761, 0.776, ...]  # her nesne için açı bilgisi
              }
        """
        self.data = data
        # Sınıf bilgilerini embedding için index'e çeviriyoruz
        # Consider adding an 'UNK' or handling potential KeyErrors if necessary
        self.class2idx = {'tank': 0, 'ship': 1} # Example classes

        # Formasyon etiketleri; nihai sınıflandırma için kullanılacak
        # Ensure this covers all possible formations in your data
        self.formation2idx = {
            "Line": 0, "Wedge": 1, "Vee": 2, "Herringbone": 3,
            "Coil": 4, "Staggered Column": 5, "Column": 6, "Echelon": 7
        }
        self.idx2formation = {v: k for k, v in self.formation2idx.items()} # Optional: for inference later

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        coords = torch.tensor(sample['coordinates'], dtype=torch.float)
        classes = torch.tensor([self.class2idx.get(c, 0) for c in sample['classes']], dtype=torch.long) # Use .get for safety
        formation_label = torch.tensor(self.formation2idx[sample['formation']], dtype=torch.long)

        # --- MODIFICATION: Add feature dimension to directions ---
        directions = torch.tensor(sample['directions'], dtype=torch.float).unsqueeze(1) # Shape: [seq_len, 1]

        # Store original length for mask generation later
        original_len = coords.shape[0]

        return coords, classes, directions, formation_label, original_len

def collate_fn(batch):
    """
    Batch içerisindeki örnekler farklı uzunlukta sequence'lere sahip olabilir.
    Bu fonksiyon, tüm örnekleri aynı uzunlukta olacak şekilde padleyip batch oluşturur.
    Ayrıca dikkat mekanizması için bir padding maskesi oluşturur.
    """
    # Unpack batch, including the original lengths
    coords_list, classes_list, directions_list, labels_list, lengths_list = zip(*batch)

    max_len = max(lengths_list)

    batch_coords, batch_classes, batch_directions = [], [], []
    # --- NEW: Create padding mask ---
    batch_padding_mask = []

    for i in range(len(coords_list)):
        coords = coords_list[i]
        classes = classes_list[i]
        directions = directions_list[i]
        original_len = lengths_list[i]

        pad_len = max_len - original_len

        if pad_len > 0:
            # Pad coordinates
            pad_coords = torch.zeros(pad_len, coords.shape[1], dtype=coords.dtype)
            coords = torch.cat([coords, pad_coords], dim=0)

            # Pad classes (Using 0, assuming mask handles it. Or use a specific padding index)
            pad_classes = torch.zeros(pad_len, dtype=classes.dtype)
            classes = torch.cat([classes, pad_classes], dim=0)

            # Pad directions (already has feature dim from __getitem__)
            pad_directions = torch.zeros(pad_len, directions.shape[1], dtype=directions.dtype)
            directions = torch.cat([directions, pad_directions], dim=0)

        batch_coords.append(coords)
        batch_classes.append(classes)
        batch_directions.append(directions)

        # --- NEW: Generate mask for this sample ---
        # Mask is True (or 1) for real tokens, False (or 0) for padding
        mask = torch.zeros(max_len, dtype=torch.bool)
        mask[:original_len] = True
        batch_padding_mask.append(mask)

    # Stack all samples into batch tensors
    batch_coords = torch.stack(batch_coords, dim=0)
    batch_classes = torch.stack(batch_classes, dim=0)
    batch_directions = torch.stack(batch_directions, dim=0)
    batch_labels = torch.stack(labels_list, dim=0) # Labels don't need padding
    batch_padding_mask = torch.stack(batch_padding_mask, dim=0)

    # The model's MultiheadAttention often expects mask where True means "ignore".
    # Let's return the inverse: True for padding, False for real tokens.
    # Check the specific requirements of your attention layer implementation.
    # Standard nn.MultiheadAttention key_padding_mask: True indicates positions that should be ignored.
    final_padding_mask = ~batch_padding_mask # Invert mask: True for padding positions

    # Return tuple: (coords, classes, directions, padding_mask, labels)
    # Note the order change: mask is often passed alongside inputs
    return batch_coords, batch_classes, batch_directions, final_padding_mask, batch_labels