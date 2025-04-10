import random

from torch.utils.data import DataLoader, random_split

def split_ata(full_dataset):
    
        
    print("split_ata working")
    
    dataset_size = len(full_dataset)
    test_size = int(0.2 * dataset_size)
    train_size = dataset_size - test_size
    # 'formation' etiketlerine göre stratified split yap
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    print("split_ata finished")
    
    return train_dataset, test_dataset




