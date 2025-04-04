import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Any # Type hinting için eklendi
import os # Dosya yolu kontrolü için

# --- Model ve Yardımcı Fonksiyonları Import Et ---
try:
    # Modelinizin GÜNCEL ve DOĞRU dosyasını import edin
    from model import HierarchicalFormationTransformer
    # Model yükleme fonksiyonunuzu import edin
    from model_tools import load_model # Bu fonksiyonun var olduğunu varsayıyoruz
except ImportError as e:
    print(f"Hata: Gerekli modüller import edilemedi: {e}")
    print("Lütfen 'model.py' ve 'model_tools.py' dosyalarının doğru yerde olduğundan emin olun.")
    exit()

# --- Sabitler ve Map'ler ---
# !!! EĞİTİMDE KULLANILANLARLA AYNI OLMALI !!!
# Sınıf ve formasyon map'lerini eğitimdekiyle tutarlı yapın.
# İdeal olarak bu map'ler eğitim sırasında kaydedilip buradan okunabilir.
CLASS_TO_IDX = {
    'tank': 0,
    'ifv': 1,
    # Eğitimde kullandığınız diğer sınıfları buraya ekleyin
    # '<PAD>': 2 # Eğer padding için özel bir indeks kullandıysanız
}
# Padding için kullanılacak indeksi belirleyin (gerekirse)
# PAD_CLASS_IDX = CLASS_TO_IDX.get('<PAD>', 0) # Eğer PAD yoksa tank (0) kullanabilir? Veya hata ver.

FORMATION_TO_IDX = {
    "Line": 0, "Wedge": 1, "Vee": 2, "Herringbone": 3,
    "Coil": 4, "Staggered Column": 5, "Column": 6, "Echelon": 7
    # Eğitimde kullandığınız tüm formasyonlar burada olmalı
}
IDX_TO_FORMATION = {i: name for name, i in FORMATION_TO_IDX.items()}

# --- MODEL HİPERPARAMETRELERİ ---
# !!! BU PARAMETRELER YÜKLENECEK MODELİN (.pth) EĞİTİLDİĞİ !!!
# !!! PARAMETRELERLE AYNI OLMALIDIR !!!
MODEL_CONFIG = {
    'class_embed_dim': 32,
    'direction_dim': 1,            # Verinize göre ayarlayın (1: açı, 2: vektör)
    'stage_dims': [256, 128],      # Önceki önerilen dengeli ayar
    'num_heads': 8,
    'num_layers': [6, 6],
    'dropout_stages': [0.2, 0.1],
    'use_adapter': True,
    'adapter_dim': 32,
    'pos_type': 'learnable',
    'max_len': 50,
    'ffn_ratio': 4,
    'coord_dim': 2,                # Genellikle 2D koordinat
    # Bu değerler map'lerden otomatik alınacak:
    # 'class_vocab_size': len(CLASS_TO_IDX),
    # 'num_formations': len(FORMATION_TO_IDX)
}

# Model ve Veri Yolları
DEFAULT_MODEL_PATH = "trained_models/hft_balanced_run_20250404_113125/best_model.pth" # Kendi run isminizi/yolunuzu yazın
DEFAULT_VAL_DATA_PATH = "dataset/val.json"


# --- Tahmin Fonksiyonu (Genellikle Değişiklik Gerekmez) ---
def predict_formation(model: HierarchicalFormationTransformer,
                      coords: List[List[float]],
                      class_tokens: List[int],
                      directions: List[float],
                      device: torch.device # Cihazı parametre olarak al
                      ) -> int:
    """
    Verilen girdilerle tek bir formasyon tahmini yapar.
    """


    # Girdileri tensor'a çevir ve batch boyutu ekle (unsqueeze(0))
    coords_t = torch.tensor(coords, dtype=torch.float32).unsqueeze(0).to(device)  # [1, N, 2]
    class_tokens_t = torch.tensor(class_tokens, dtype=torch.long).unsqueeze(0).to(device)  # [1, N]
    # Directions: [N] -> [N, 1] -> [1, N, 1] (Modelin beklediği direction_dim=1 ise)
    # Eğer direction_dim=2 ise .unsqueeze(-1) yapmayın, direkt [N, 2] -> [1, N, 2] olsun
    if MODEL_CONFIG['direction_dim'] == 1:
        directions_t = torch.tensor(directions, dtype=torch.float32).unsqueeze(-1).unsqueeze(0).to(device)
    elif MODEL_CONFIG['direction_dim'] == 2:
         directions_t = torch.tensor(directions, dtype=torch.float32).unsqueeze(0).to(device) # Shape [1, N, 2] olmalı
    else:
        raise ValueError(f"Unsupported direction_dim: {MODEL_CONFIG['direction_dim']}")


    with torch.no_grad(): # Gradyan hesaplamayı kapat
        # Padding maskesi tek örnek için None
        output_logits = model(coords=coords_t,
                              class_tokens=class_tokens_t,
                              directions=directions_t,
                              key_padding_mask=None)
        prediction_idx = torch.argmax(output_logits, dim=-1).item()

    return prediction_idx

# --- Ana Fonksiyon ---
def main(model_path: str, val_data_path: str):
    # Cihazı belirle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan cihaz: {device}")

    # --- Modeli Oluştur ---
    print("Model mimarisi oluşturuluyor...")
    # Map'lerden gerçek boyutları al
    actual_class_vocab_size = len(CLASS_TO_IDX)
    actual_num_formations = len(FORMATION_TO_IDX)

    # ModelConfig'i kullanarak modeli oluştur
    model = HierarchicalFormationTransformer(
        num_formations=actual_num_formations,
        class_vocab_size=actual_class_vocab_size,
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

    # Modeli cihaza taşı (ağırlıkları yüklemeden ÖNCE)
    model.to(device)

    # --- Model Ağırlıklarını Yükle ---
    if not os.path.exists(model_path):
        print(f"HATA: Model dosyası bulunamadı: {model_path}")
        print(f"Lütfen DEFAULT_MODEL_PATH değişkenini veya komut satırı argümanını kontrol edin.")
        return

    print(f"Model ağırlıkları yükleniyor: {model_path}")
    try:
        # load_model fonksiyonunuzun modeli, yolu ve cihazı alıp state_dict'i yüklediğini varsayalım
        # Eğer fonksiyonunuz device almıyorsa, state_dict'i map_location ile yüklemeniz gerekebilir.
        load_model(model_path, model)
        print("Model başarıyla yüklendi.")
    except Exception as e:
        print(f"Model yüklenirken hata oluştu: {e}")
        print("Model mimarisi (MODEL_CONFIG) ile yüklenen dosyanın parametrelerinin eşleştiğinden emin olun.")
        return # Hata durumunda çık

    # --- Veriyi Oku ve Hazırla ---
    if not os.path.exists(val_data_path):
        print(f"HATA: Veri dosyası bulunamadı: {val_data_path}")
        return

    print(f"Doğrulama verisi okunuyor: {val_data_path}")
    try:
        df = pd.read_json(val_data_path)
    except Exception as e:
        print(f"JSON dosyası okunamadı: {val_data_path} - Hata: {e}")
        return

    print("Veri ön işleniyor...")
    # Sınıf isimlerini indekslere çevir
    # Bilinmeyen sınıf varsa hata vermek yerine uyarı verip 0 atayalım (veya PAD_CLASS_IDX)
    def classes_to_indices(class_list):
        indices = []
        for c in class_list:
            idx = CLASS_TO_IDX.get(c)
            if idx is None:
                print(f"Uyarı: Bilinmeyen sınıf '{c}' bulundu, 0 olarak atanıyor.")
                idx = 0 # Veya PAD_CLASS_IDX varsa onu kullanın
            indices.append(idx)
        return indices

    df['class_tokens'] = df['classes'].apply(classes_to_indices)

    # Formasyon isimlerini indekslere çevir (-1: bilinmeyen/geçersiz)
    df['formation_idx'] = df['formation'].apply(lambda x: FORMATION_TO_IDX.get(x, -1)).astype(int)

    # directions sütununun varlığını kontrol et
    if 'directions' not in df.columns:
        print("HATA: 'directions' sütunu JSON dosyasında bulunamadı.")
        return
    if 'coordinates' not in df.columns:
         print("HATA: 'coordinates' sütunu JSON dosyasında bulunamadı.")
         return

    # --- Tahminleri Yap ---
    predicted_indices = []
    original_indices = []
    problematic_indices = [] # Tahmin yapılamayan satırları takip et
    

    print("Tahminler yapılıyor...")
    for index, row in df.iterrows():
        try:
            coordinates = row['coordinates']
            class_tokens = row['class_tokens']
            directions = row['directions']
            formation_idx = row['formation_idx']

            # Temel girdi kontrolleri
            if not isinstance(coordinates, list) or not isinstance(class_tokens, list) or not isinstance(directions, list):
                raise ValueError("Geçersiz veri tipi (liste değil)")
            if not coordinates or not class_tokens or not directions: # Boş listeler
                 raise ValueError("Boş girdi listesi")
            if not (len(coordinates) == len(class_tokens) == len(directions)):
                 raise ValueError(f"Uzunluklar eşleşmiyor: C({len(coordinates)}) Cl({len(class_tokens)}) D({len(directions)})")
            if len(coordinates) > MODEL_CONFIG['max_len']:
                print(f"Uyarı: Satır {index} uzunluğu ({len(coordinates)}) > max_len ({MODEL_CONFIG['max_len']}). Kırpma yapılmadı!")
                # İsterseniz burada kırpma ekleyebilirsiniz:
                # coordinates = coordinates[:MODEL_CONFIG['max_len']]
                # class_tokens = class_tokens[:MODEL_CONFIG['max_len']]
                # directions = directions[:MODEL_CONFIG['max_len']]

            # Tahmini yap (cihaz bilgisini de gönder)
            prediction = predict_formation(model, coordinates, class_tokens, directions, device)

            original_indices.append(formation_idx)
            predicted_indices.append(prediction)

        except Exception as e:
            print(f"HATA: Satır {index} işlenirken hata oluştu: {e}. Bu satır atlanıyor.")
            problematic_indices.append(index)
            # Hata durumunda listelerin senkronize kalması için placeholder ekle (isteğe bağlı)
            # original_indices.append(-2) # Hata kodu
            # predicted_indices.append(-2)

    print(f"Tahminler tamamlandı. {len(problematic_indices)} satırda hata oluştu.")

    # --- Sonuçları Değerlendir ---
    if not predicted_indices:
        print("Hiçbir tahmin yapılamadı.")
        return

    predicted_array = np.array(predicted_indices)
    original_array = np.array(original_indices)
    
    print(predicted_array)
    print(original_array)

    # Sadece geçerli orijinal etiketlere sahip olanları ve hatasız işlenenleri değerlendir
    # (Eğer hata durumunda placeholder eklemediyseniz, orijinal_array'i filtrelemek yeterli)
    valid_mask = original_array != -1
    if len(valid_mask) != len(predicted_array):
         print("Uyarı: Tahmin ve orijinal etiket sayıları farklı, sadece eşleşenler değerlendirilecek.")
         # Boyutları eşitlemek için daha karmaşık bir mantık gerekebilir, şimdilik basit tutalım.
         min_len = min(len(valid_mask), len(predicted_array))
         valid_mask = valid_mask[:min_len]
         predicted_array = predicted_array[:min_len]
         original_array = original_array[:min_len]


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

    # İsteğe bağlı: Daha detaylı raporlama
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        target_names = [IDX_TO_FORMATION.get(i, f'Bilinmeyen({i})') for i in sorted(FORMATION_TO_IDX.values())]
        # Sadece veri setinde bulunan sınıflar için rapor oluştur
        unique_labels_in_data = np.unique(np.concatenate((valid_originals, valid_predictions)))
        filtered_target_names = [IDX_TO_FORMATION.get(i, f'Bilinmeyen({i})') for i in sorted(unique_labels_in_data) if i in IDX_TO_FORMATION]
        print("\nSınıflandırma Raporu:")
        print(classification_report(valid_originals, valid_predictions, target_names=filtered_target_names, labels=sorted(unique_labels_in_data), zero_division=0))

        # Confusion matrix için de etiketleri belirle
        all_labels = sorted(list(set(valid_originals) | set(valid_predictions)))
        cm = confusion_matrix(valid_originals, valid_predictions, labels=all_labels)
        print("\nConfusion Matrix:")
        print(f"Etiketler: {[IDX_TO_FORMATION.get(i, f'Bilinmeyen({i})') for i in all_labels]}")
        print(cm)

    except ImportError:
        print("\nDetaylı raporlama için 'scikit-learn' kütüphanesi gerekli: pip install scikit-learn")
    except Exception as report_error:
        print(f"\nRaporlama sırasında bir hata oluştu: {report_error}")


if __name__ == "__main__":
    # Komut satırı argümanları yerine varsayılan yolları kullan
    # İsterseniz argparse ekleyerek bu yolları dışarıdan alabilirsiniz
    model_to_load = DEFAULT_MODEL_PATH
    validation_data = DEFAULT_VAL_DATA_PATH

    print(f"Kullanılacak Model Yolu: {model_to_load}")
    print(f"Kullanılacak Veri Yolu: {validation_data}")

    main(model_path=model_to_load, val_data_path=validation_data)