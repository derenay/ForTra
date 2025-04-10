import torch
import pandas as pd # <- DataFrame okumak için hala gerekli
import numpy as np
from typing import List, Dict, Any # Type hinting için eklendi
import os # Dosya yolu kontrolü için
import json # JSON okumak için eklendi

# DataLoader import'u ekleniyor
from torch.utils.data import DataLoader

# --- Model ve Yardımcı Fonksiyonları Import Et ---
try:
    # Modelinizin GÜNCEL ve DOĞRU dosyasını import edin
    from model import HierarchicalFormationTransformer
    # Model yükleme fonksiyonunuzu import edin
    from model_tools import load_model # Bu fonksiyonun var olduğunu varsayıyoruz
   
    from dataset import FormationDataset, collate_fn
except ImportError as e:
    print(f"Hata: Gerekli modüller import edilemedi: {e}")
    print("Lütfen 'model.py', 'model_tools.py' ve 'dataset.py' dosyalarının doğru yerde olduğundan emin olun.")
    exit()

# --- Sabitler ve Map'ler ---
# !!! EĞİTİMDE KULLANILANLARLA AYNI OLMALI !!!
# Bu map'ler dataset.py içindekiyle tutarlı olmalı veya oradan alınmalı
CLASS_TO_IDX = {
    'tank': 0,
    'ifv': 1, # <- dataset.py'dekiyle aynı olmalı
    # Diğer sınıflar...
}
FORMATION_TO_IDX = {
    "Line": 0, "Wedge": 1, "Vee": 2, "Herringbone": 3,
    "Coil": 4, "Staggered Column": 5, "Column": 6, "Echelon": 7
    # Diğer formasyonlar...
}
IDX_TO_FORMATION = {i: name for name, i in FORMATION_TO_IDX.items()}

# --- MODEL HİPERPARAMETRELERİ ---
# !!! BU PARAMETRELER YÜKLENECEK MODELİN (.pth) EĞİTİLDİĞİ !!!
# !!! PARAMETRELERLE AYNI OLMALIDIR !!!
MODEL_CONFIG = {
    'class_embed_dim': 32,
    'direction_dim': 1,
    'stage_dims': [256, 128], # Önceki önerilen dengeli ayar
    'num_heads': 8,
    'num_layers': [6, 6],
    'dropout_stages': [0.2, 0.1],
    'use_adapter': True,
    'adapter_dim': 32,
    'pos_type': 'learnable',
    'max_len': 50, # Eğitimdeki config ile aynı olmalı
    'ffn_ratio': 4,
    'coord_dim': 2,
}

# Model ve Veri Yolları
DEFAULT_MODEL_PATH = "trained_models/hft_balanced_run_20250410_114415/best_model copy 3.pth" # Kendi run isminizi/yolunuzu yazın
DEFAULT_VAL_DATA_PATH = "dataset/val.json"
DEFAULT_BATCH_SIZE = 32 # Doğrulama için batch boyutu


# --- Ana Fonksiyon (DataLoader Kullanacak Şekilde Güncellendi) ---
def main(model_path: str, val_data_path: str, batch_size: int):
    # Cihazı belirle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan cihaz: {device}")

    # --- Modeli Oluştur ---
    print("Model mimarisi oluşturuluyor...")
    actual_class_vocab_size = len(CLASS_TO_IDX)
    actual_num_formations = len(FORMATION_TO_IDX)

    model = HierarchicalFormationTransformer(
        num_formations=8,
        class_vocab_size=2,
        class_embed_dim=MODEL_CONFIG['class_embed_dim'],
        direction_dim=MODEL_CONFIG['direction_dim'],
        coord_dim=MODEL_CONFIG['coord_dim'],
        stage_dims=MODEL_CONFIG['stage_dims'],
        num_heads=MODEL_CONFIG['num_heads'],
        num_layers=MODEL_CONFIG['num_layers'],
        dropout_stages=MODEL_CONFIG['dropout_stages'],
        use_adapter=MODEL_CONFIG['use_adapter'],
        adapter_dim=MODEL_CONFIG['adapter_dim'],
        pos_type=MODEL_CONFIG['pos_type'],
        max_len=MODEL_CONFIG['max_len'],
        ffn_ratio=MODEL_CONFIG['ffn_ratio']
    )
    model.to(device)

    # --- Model Ağırlıklarını Yükle ---
    if not os.path.exists(model_path):
        print(f"HATA: Model dosyası bulunamadı: {model_path}")
        return
    print(f"Model ağırlıkları yükleniyor: {model_path}")
    try:
        load_model(model_path, model)
        print("Model başarıyla yüklendi.")
    except Exception as e:
        print(f"Model yüklenirken hata oluştu: {e}")
        print("Model mimarisi (MODEL_CONFIG) ile yüklenen dosyanın parametrelerinin eşleştiğinden emin olun.")
        return

    # --- Veriyi Oku ve Hazırla (Dataset ve DataLoader ile) ---
    if not os.path.exists(val_data_path):
        print(f"HATA: Veri dosyası bulunamadı: {val_data_path}")
        return

    print(f"Doğrulama verisi okunuyor ve Dataset oluşturuluyor: {val_data_path}")
    try:
        # Pandas yerine json modülü ile direkt liste olarak okuyalım
        with open(val_data_path, 'r') as f:
            data_samples = json.load(f)
        # FormationDataset'i oluştur (kanonik sıralama ve indeksleme burada yapılacak)
        val_dataset = FormationDataset(data_samples)
        # Önemli: Dataset içindeki map'lerin yukarıdakilerle tutarlı olduğunu varsayıyoruz
        # Eğer değilse, dataset.class2idx ve dataset.formation2idx kullanılmalı
        if len(val_dataset.class2idx) != actual_class_vocab_size or \
           len(val_dataset.formation2idx) != actual_num_formations:
            print("Uyarı: val.py içindeki map'ler ile Dataset içindeki map'ler farklı boyutlarda!")
            print(f"  val.py Sınıf Sayısı: {actual_class_vocab_size}, Dataset Sınıf Sayısı: {len(val_dataset.class2idx)}")
            print(f"  val.py Formasyon Sayısı: {actual_num_formations}, Dataset Formasyon Sayısı: {len(val_dataset.formation2idx)}")
            # İsterseniz burada dataset'ten alınanları kullanabilirsiniz:
            # actual_class_vocab_size = len(val_dataset.class2idx)
            # actual_num_formations = len(val_dataset.formation2idx)
            # print("Uyarı: Dataset içindeki boyutlar kullanılacak.")


    except Exception as e:
        print(f"Veri okuma veya Dataset oluşturma hatası: {e}")
        return

    print("DataLoader oluşturuluyor...")
    # shuffle=False önemlidir, validasyon sırası önemli değildir ve tekrar edilebilirliği artırır
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            collate_fn=collate_fn, # Padding ve batch'leme için
                            num_workers=max(0, os.cpu_count() // 2)) # Num worker ayarı

    # --- Tahminleri Yap (Toplu Olarak) ---
    model.eval() # Modeli inference moduna al
    all_predictions = []
    all_labels = []

    print(f"Tahminler yapılıyor (Batch Size: {batch_size})...")
    with torch.no_grad(): # Gradyan hesaplamayı kapat
        for batch_idx, (coords, classes, directions, padding_mask, labels) in enumerate(val_loader):
            # Veriyi cihaza taşı
            coords, classes, directions, padding_mask, labels = \
                coords.to(device), classes.to(device), directions.to(device), padding_mask.to(device), labels.to(device)

            # Model ile toplu tahmin yap
            outputs = model(coords, classes, directions, key_padding_mask=padding_mask)

            # Tahminleri al (logit'lerden en yüksek olasılıklı sınıfın indeksi)
            _, predicted_indices_batch = torch.max(outputs, 1)

            # Tahminleri ve etiketleri CPU'ya alıp listelerde topla
            all_predictions.extend(predicted_indices_batch.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # İlerleme göstergesi (isteğe bağlı)
            if (batch_idx + 1) % 10 == 0:
                 print(f"  Batch {batch_idx+1}/{len(val_loader)} tamamlandı.", end='\r')

    print("\nTahminler tamamlandı.")

    # --- Sonuçları Değerlendir ---
    if not all_predictions:
        print("Hiçbir tahmin yapılamadı.")
        return

    predicted_array = np.array(all_predictions)
    original_array = np.array(all_labels)

    # Geçerli etiketleri filtrele (eğer dataset -1 döndürdüyse)
    valid_mask = original_array != -1
    valid_predictions = predicted_array[valid_mask]
    valid_originals = original_array[valid_mask]

    if len(valid_originals) == 0:
        print("Değerlendirme için geçerli örnek bulunamadı.")
        return

    total_valid_samples = len(valid_originals)
    correct_predictions = np.sum(valid_predictions == valid_originals)
    accuracy = (correct_predictions / total_valid_samples) * 100 if total_valid_samples > 0 else 0

    print(f"\n--- Değerlendirme Sonuçları ({val_data_path}) ---")
    print(f"Toplam Geçerli Örnek Sayısı: {total_valid_samples}")
    print(f"Doğru Tahmin Sayısı: {correct_predictions}")
    print(f"Doğruluk (Accuracy): {accuracy:.2f}%")
    print(valid_originals)
    print(valid_predictions)
   
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        target_names = [IDX_TO_FORMATION.get(i, f'Bilinmeyen({i})') for i in sorted(FORMATION_TO_IDX.values())]
        unique_labels_in_data = np.unique(np.concatenate((valid_originals, valid_predictions)))
        # Rapor ve matris için etiketleri dataset'ten gelenlere göre filtrele
        filtered_target_names = [IDX_TO_FORMATION.get(i, f'Bilinmeyen({i})') for i in sorted(unique_labels_in_data) if i in IDX_TO_FORMATION]
        report_labels = [i for i in sorted(unique_labels_in_data) if i in IDX_TO_FORMATION] # Sadece bilinen etiketler

        if not report_labels:
             print("\nRaporda gösterilecek geçerli etiket bulunamadı.")
        else:
            print("\nSınıflandırma Raporu:")
            print(classification_report(valid_originals, valid_predictions, target_names=filtered_target_names, labels=report_labels, zero_division=0))
            
            # Confusion matrix
            cm = confusion_matrix(valid_originals, valid_predictions, labels=report_labels)
            print("\nConfusion Matrix:")
            print(f"Etiketler: {filtered_target_names}")
            print(cm)

    except ImportError:
        print("\nDetaylı raporlama için 'scikit-learn' kütüphanesi gerekli: pip install scikit-learn")
    except Exception as report_error:
        print(f"\nRaporlama sırasında bir hata oluştu: {report_error}")


if __name__ == "__main__":
    # Model ve veri yollarını buradan ayarlayın
    model_to_load = DEFAULT_MODEL_PATH
    validation_data = DEFAULT_VAL_DATA_PATH
    batch_size_val = DEFAULT_BATCH_SIZE

    # Yolun var olup olmadığını kontrol et (opsiyonel ama iyi pratik)
    if 'YYYYMMDD_HHMMSS' in model_to_load:
        print(f"Lütfen DEFAULT_MODEL_PATH değişkenindeki '{model_to_load}' yolunu geçerli bir model yoluyla güncelleyin.")
    else:
        print(f"Kullanılacak Model Yolu: {model_to_load}")
        print(f"Kullanılacak Veri Yolu: {validation_data}")
        print(f"Kullanılacak Batch Boyutu: {batch_size_val}")
        main(model_path=model_to_load, val_data_path=validation_data, batch_size=batch_size_val)