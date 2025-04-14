import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict, Any # Gerekli olmayabilir ama iyi pratik

# Veri setini oku
# JSON dosyasının adının doğru olduğundan emin olun
try:
    df = pd.read_json("tank_formations_mixed.json")
    print(f"Veri başarıyla okundu: karistirilmis_veri.json ({len(df)} örnek)")
except Exception as e:
    print(f"Hata: JSON dosyası okunamadı: {e}")
    exit()

# Her formasyon için çizim yap
# Örnek sayısını sınırlamak isterseniz: for ... in zip(...[:10]): # İlk 10 örneği çiz
for index, (formation, coordinates, classes, angles) in enumerate(
    zip(df['formation'], df['coordinates'], df['classes'], df['directions'])
):

    print(f"\nİşleniyor: Örnek {index}, Formasyon: {formation}")

    # --- Kanonik Sıralama Mantığını Uygula ---
    # 1. Verileri birleştir (sınıf bilgisi sıralamayı etkilemez ama hizalamak için lazım)
    if not isinstance(coordinates, list) or \
       not isinstance(classes, list) or \
       not isinstance(angles, list) or \
       not coordinates:
           print(f" Uyarı: Örnek {index} geçersiz veya boş veri içeriyor, atlanıyor.")
           continue

    if not (len(coordinates) == len(classes) == len(angles)):
        print(f" Uyarı: Örnek {index} için uzunluklar uyuşmuyor, atlanıyor.")
        continue

    combined = list(zip(coordinates, classes, angles))

    # 2. Koordinatlara göre sırala ([x, y] çiftlerine göre)
    try:
        # Orijinal sırayı koruyarak yeni bir sıralanmış liste oluştur
        sorted_combined = sorted(combined, key=lambda item: item[0])
    except Exception as e:
        print(f" Hata: Örnek {index} sıralanırken hata: {e}. Bu örnek atlanıyor.")
        continue

    # 3. Sıralanmış verileri ayır
    if not sorted_combined:
        print(f" Uyarı: Örnek {index} işlendikten sonra boş kaldı, atlanıyor.")
        continue

    sorted_coords, sorted_classes, sorted_angles = zip(*sorted_combined)
    # zip tuple döndürür, liste yapalım
    sorted_coords_list = list(sorted_coords)
    sorted_angles_list = list(sorted_angles)

    # Orijinal koordinatları da ayıralım (çizgi için)
    original_coords_list = coordinates # Zaten liste

    # --- Çizim ---
    plt.figure(figsize=(10, 8)) # Boyutu biraz büyütelim

    # X ve Y değerlerini al (Sıralanmış koordinatlardan)
    x_values_sorted = [coord[0] for coord in sorted_coords_list]
    y_values_sorted = [coord[1] for coord in sorted_coords_list]

    # Orijinal sıraya göre X ve Y (kırmızı çizgi için)
    x_values_orig = [coord[0] for coord in original_coords_list]
    y_values_orig = [coord[1] for coord in original_coords_list]

    # Tankları noktalarla çiz (sıralanmış koordinatları kullanalım)
    plt.scatter(x_values_sorted, y_values_sorted, color='blue', label='Tanklar (Sıralı Konum)', zorder=3)

    # Orijinal sırayı gösteren çizgiyi çiz
    plt.plot(x_values_orig, y_values_orig, color='red', linestyle=':', alpha=0.6, label=f'Orijinal Sıra Bağlantısı')

    # Ok çizimi için ayarlar
    arrow_len = 0.04 # Ok boyutunu ayarlayabilirsiniz

    # Sıralanmış tankların üzerine sıralama indekslerini yazdır ve okları çiz
    for i, (x, y, angle_norm) in enumerate(zip(x_values_sorted, y_values_sorted, sorted_angles_list)):
        # Sıralama indeksini yazdır
        plt.text(x + 0.005, y + 0.005, str(i), color='black', fontsize=9, zorder=4) # Etiketleri noktaların biraz sağına/üstüne

        # Okları çiz
        angle_rad = angle_norm * 2 * np.pi
        dx = arrow_len * np.cos(angle_rad)
        dy = arrow_len * np.sin(angle_rad)
        plt.arrow(x, y, dx, dy, head_width=0.01, head_length=0.015, fc='green', ec='green', zorder=2)

    plt.title(f"Formasyon: {formation} (Sıralama Sonrası İndeksler Gösteriliyor)")
    plt.xlabel("X Koordinatları")
    plt.ylabel("Y Koordinatları")
    plt.legend()
    plt.grid(True)
    # Eksen limitlerini ayarlayarak daha iyi görünüm elde edilebilir (isteğe bağlı)
    # plt.xlim(min(x_values_sorted)-0.1, max(x_values_sorted)+0.1)
    # plt.ylim(min(y_values_sorted)-0.1, max(y_values_sorted)+0.1)
    plt.gca().set_aspect('equal', adjustable='box') # Eksenleri eşit ölçekle
    plt.show()

print("\nTüm örnekler için çizim tamamlandı.")