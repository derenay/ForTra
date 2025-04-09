# mixer_no_args.py
# Date: 2025-04-09
# Açıklama: Komut satırı argümanları olmadan çalışan versiyon.
# Dosya yolları ve karıştırma sayısı kod içinden ayarlanır.

import json
import random
import os
import sys


# --- AYARLAR ---
# Lütfen aşağıdaki değişkenleri kendi dosya yollarınız ve istediğiniz
# karıştırma sayısıyla güncelleyin.
INPUT_FILENAME = "tank_formations_mixed.json"  # Okunacak orijinal JSON dosyasının adı
OUTPUT_FILENAME = "karistirilmis_veri.json" # Oluşturulacak yeni JSON dosyasının adı
NUM_SHUFFLES_PER_SAMPLE = 10 # Orijinal her örnek için kaç adet karıştırılmış kopya oluşturulacak?
# -------------

def shuffle_sample_elements(sample):
    """
    Verilen bir formasyon örneğindeki (sample) tankların sırasını karıştırır.
    Koordinatlar, sınıflar ve yönler arasındaki eşleşmeyi korur.

    Args:
        sample (dict): Tek bir formasyon örneği içeren sözlük.
                       {'coordinates': [...], 'classes': [...], 'directions': [...], 'formation': '...'}
                       formatında olmalıdır.

    Returns:
        dict: Elemanlarının sırası karıştırılmış yeni bir örnek sözlüğü veya
              girdi geçersizse None.
    """
    try:
        # Gerekli anahtarların varlığını kontrol et
        if not all(k in sample for k in ['coordinates', 'classes', 'directions', 'formation']):
             # print(f"Uyarı: Örnekte beklenen anahtarlar eksik, atlanıyor: {list(sample.keys())}")
             return None

        coords = sample['coordinates']
        classes = sample['classes']
        directions = sample['directions']
        formation_label = sample['formation'] # Formasyon etiketini koru

        # Veri tiplerini kontrol et (liste olmalılar)
        if not isinstance(coords, list) or not isinstance(classes, list) or not isinstance(directions, list):
             # print(f"Uyarı: Örnek içindeki veriler liste değil, atlanıyor.")
             return None

        num_tanks = len(coords)

        # Listelerin boş olup olmadığını ve uzunluklarının eşleşip eşleşmediğini kontrol et
        if num_tanks == 0:
            # print("Uyarı: Boş koordinat listesi, örnek atlanıyor.")
            return None
        if not (len(classes) == num_tanks and len(directions) == num_tanks):
            # print(f"Uyarı: Liste uzunlukları tutarsız (Coords: {num_tanks}, Classes: {len(classes)}, Dirs: {len(directions)}), örnek atlanıyor.")
            return None

        # Karıştırmak için indis listesi oluştur
        indices = list(range(num_tanks))
        random.shuffle(indices) # İndisleri yerinde karıştır

        # Karıştırılmış indisleri kullanarak yeni listeler oluştur
        shuffled_coords = [coords[i] for i in indices]
        shuffled_classes = [classes[i] for i in indices]
        shuffled_directions = [directions[i] for i in indices]

        # Yeni (karıştırılmış) örneği oluştur
        new_sample = {
            "coordinates": shuffled_coords,
            "classes": shuffled_classes,
            "formation": formation_label, # Orijinal etiketi kullan
            "directions": shuffled_directions
        }
        return new_sample

    except KeyError as e:
        # print(f"Uyarı: Örnekte beklenen anahtar bulunamadı ('{e}'), atlanıyor.")
        return None
    except TypeError as e:
        # print(f"Uyarı: Örnek içindeki verilerde tip hatası ('{e}'), atlanıyor.")
        return None
    except Exception as e:
        # print(f"Uyarı: Örnek işlenirken beklenmedik hata ('{e}'), atlanıyor.")
        return None


def augment_data_with_shuffling(input_filepath, output_filepath, num_shuffles_per_sample):
    """
    Giriş JSON dosyasındaki her bir formasyon örneği için belirtilen sayıda
    karıştırılmış (shuffled) kopya oluşturur ve yeni bir JSON dosyasına kaydeder.
    Argümanlar doğrudan kod içinden alınır.
    """
    if num_shuffles_per_sample <= 0:
        print("Hata: Karıştırma sayısı (NUM_SHUFFLES_PER_SAMPLE) pozitif bir tamsayı olmalıdır.")
        return

    # Giriş dosyasını oku
    try:
        print(f"'{input_filepath}' dosyasından veri okunuyor...")
        if not os.path.exists(input_filepath):
            raise FileNotFoundError(f"Giriş dosyası bulunamadı: '{input_filepath}'")
        with open(input_filepath, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        if not isinstance(original_data, list):
            raise TypeError("JSON dosyasının içeriği bir liste olmalıdır.")
        print(f"{len(original_data)} adet orijinal örnek okundu.")
    except FileNotFoundError as e:
        print(f"Hata: {e}")
        sys.exit(1) # Programdan çık
    except json.JSONDecodeError as e:
        print(f"Hata: JSON dosyası çözümlenemedi: {e}")
        sys.exit(1)
    except TypeError as e:
        print(f"Hata: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Giriş dosyası okunurken beklenmedik bir hata oluştu: {e}")
        sys.exit(1)

    augmented_data = []
    processed_count = 0
    skipped_count = 0

    print(f"Her orijinal örnek için {num_shuffles_per_sample} adet karıştırılmış kopya oluşturuluyor...")

    # Her orijinal örnek için karıştırma işlemi yap
    for i, original_sample in enumerate(original_data):
        # Belirtilen sayıda karıştırılmış kopya oluştur
        shuffles_created_for_sample = 0
        for _ in range(num_shuffles_per_sample):
            # Karıştırma fonksiyonunu çağır
            # Orijinal örneği her seferinde kullanmak önemli, deepcopy gereksiz
            shuffled_sample = shuffle_sample_elements(original_sample)

            # Eğer karıştırma başarılıysa listeye ekle
            if shuffled_sample:
                augmented_data.append(shuffled_sample)
                shuffles_created_for_sample += 1

        if shuffles_created_for_sample > 0:
             processed_count += 1
        # Boş olmayan ama işlenemeyen örnekler için uyarı ver
        elif isinstance(original_sample.get('coordinates'), list) and len(original_sample['coordinates']) > 0:
             skipped_count += 1
             print(f"Uyarı: Orijinal örnek #{i+1} için hiç geçerli karışık kopya oluşturulamadı (muhtemelen format hatası).")


    print(f"\nİşlem tamamlandı.")
    print(f"İşlenen orijinal örnek sayısı: {processed_count}")
    if skipped_count > 0:
        print(f"Atlanan (hatalı/boş) orijinal örnek sayısı: {skipped_count}")
    print(f"Toplam oluşturulan artırılmış (karıştırılmış) örnek sayısı: {len(augmented_data)}")

    # Artırılmış veriyi yeni dosyaya yaz
    try:
        print(f"Artırılmış veri '{output_filepath}' dosyasına yazılıyor...")
        # Çıktı dizininin var olup olmadığını kontrol et, yoksa oluştur
        output_dir = os.path.dirname(output_filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Oluşturuldu: Çıktı dizini '{output_dir}'")

        with open(output_filepath, 'w', encoding='utf-8') as f:
            # indent=2 okunabilirliği artırır, ensure_ascii=False Türkçe karakterleri korur
            json.dump(augmented_data, f, indent=2, ensure_ascii=False)
        print("Veri başarıyla kaydedildi.")

    except IOError as e:
        print(f"Hata: Çıktı dosyası yazılamadı: {e}")
    except Exception as e:
        print(f"Çıktı dosyası yazılırken beklenmedik bir hata oluştu: {e}")

# Doğrudan çalıştırma kısmı (argparse olmadan)
if __name__ == "__main__":
    # Kodun başındaki ayarlardan değerleri al
    input_file = INPUT_FILENAME
    output_file = OUTPUT_FILENAME
    num_shuffles = NUM_SHUFFLES_PER_SAMPLE

    print("--- Veri Karıştırma Başlatılıyor ---")
    print(f"Giriş Dosyası: {input_file}")
    print(f"Çıktı Dosyası: {output_file}")
    print(f"Örnek Başına Karışık Kopya Sayısı: {num_shuffles}")
    print("------------------------------------")

    # Ana fonksiyonu doğrudan tanımlanan değerlerle çağır
    augment_data_with_shuffling(input_file, output_file, num_shuffles)

    print("\n--- Veri Karıştırma Tamamlandı ---")