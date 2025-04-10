import numpy as np
import random
import math
import json

# --- Diğer generate fonksiyonları (generate_herringbone vb.) burada tanımlı olabilir ---

def generate_staggered_column(
    num_units_range=(4, 12),       # Formasyondaki min/max birim sayısı
    axis_direction_norm=None,    # Formasyonun ana yönü (0-1 arası, None ise rastgele)
    unit_sep_mean=0.08,          # Eksen boyunca birimler arası ORTALAMA mesafe
    # unit_sep_std=0.0,           # <<< Kaldırıldı/Sıfırlandı
    offset_dist_mean=0.04,       # Ekseninden sağa/sola ORTALAMA kayma mesafesi
    # offset_dist_std=0.0,        # <<< Kaldırıldı/Sıfırlandı
    coord_noise_scale=0.0,      # <<< GÜRÜLTÜSÜZ: Varsayılan 0
    direction_noise_scale=0.0,  # <<< GÜRÜLTÜSÜZ: Varsayılan 0
    center_x_range=(0.1, 0.9),
    center_y_range=(0.1, 0.9),
    # --- Değişkenlik Faktörleri ---
    sep_variation_factor=0.4,   # Mesafe, ortalamanın +/- %40'ı kadar değişebilir
    offset_variation_factor=0.5 # Kayma, ortalamanın +/- %50'si kadar değişebilir
) -> dict:
    """
    Belirtilen parametrelere göre GÜRÜLTÜSÜZ ancak birimler arası mesafe ve
    eksenden kayma miktarı DEĞİŞKEN olan Staggered Column formasyonları üretir.
    coord_noise_scale ve direction_noise_scale 0 ise gürültüsüz olur.
    """

    num_units = random.randint(num_units_range[0], num_units_range[1])
    if num_units < 2: num_units = 2

    center_x = random.uniform(center_x_range[0], center_x_range[1])
    center_y = random.uniform(center_y_range[0], center_y_range[1])

    if axis_direction_norm is None:
        axis_direction_norm = random.random()
    axis_angle_rad = axis_direction_norm * 2 * math.pi

    coordinates = []
    directions_rad = []

    axis_dx = math.cos(axis_angle_rad)
    axis_dy = math.sin(axis_angle_rad)
    perp_dx = -axis_dy
    perp_dy = axis_dx

    current_pos_x = center_x
    current_pos_y = center_y

    for i in range(num_units):
        # Birimler arası mesafeyi belirle (DEĞİŞKEN AMA GÜRÜLTÜSÜZ)
        sep_low = unit_sep_mean * (1 - sep_variation_factor)
        sep_high = unit_sep_mean * (1 + sep_variation_factor)
        # Her adımda bu aralıktan rastgele bir mesafe seç
        sep = random.uniform(max(0.01, sep_low), max(0.02, sep_high))

        if i > 0:
            current_pos_x += axis_dx * sep
            current_pos_y += axis_dy * sep

        # Sağa/sola kayma mesafesini belirle (DEĞİŞKEN AMA GÜRÜLTÜSÜZ)
        offset_low = offset_dist_mean * (1 - offset_variation_factor)
        offset_high = offset_dist_mean * (1 + offset_variation_factor)
        # Her adımda bu aralıktan rastgele bir kayma miktarı seç
        offset_dist = random.uniform(max(0.005, offset_low), max(0.01, offset_high)) # Minik offsetlere izin ver

        # Sağa veya sola kaydır
        unit_x = current_pos_x
        unit_y = current_pos_y
        if i % 2 == 1: # Tek indeksliler sağa
            unit_x -= perp_dx * offset_dist
            unit_y -= perp_dy * offset_dist
        else: # Çift indeksliler sola
            unit_x += perp_dx * offset_dist
            unit_y += perp_dy * offset_dist

        # Koordinatlara gürültü ekle (coord_noise_scale=0 ise eklenmez)
        final_x = unit_x + np.random.normal(0, coord_noise_scale)
        final_y = unit_y + np.random.normal(0, coord_noise_scale)

        # YÖN HESAPLAMA: Ana eksen yönü + gürültü (direction_noise_scale=0 ise eklenmez)
        final_direction_rad = axis_angle_rad + np.random.normal(0, direction_noise_scale * 2 * math.pi)

        coordinates.append([final_x, final_y])
        directions_rad.append(final_direction_rad)

    # Radyan yönleri 0-1 arasına normalize et
    directions_norm = [(angle / (2 * math.pi)) % 1.0 for angle in directions_rad]

    result = {
        "coordinates": coordinates,
        "classes": ["tank"] * num_units,
        "formation": "Staggered Column",
        "directions": directions_norm
    }
    return result

# --- Örnek Kullanım ---
if __name__ == "__main__":
    samples_per_formation = 100
    formation_generators = {
        # Diğer formasyonlar için de gürültüsüz hale getirilmiş fonksiyonlar...
        # "Herringbone": generate_herringbone_noiseless,
        # "Column": generate_column_noiseless,
        "Staggered Column": generate_staggered_column, # Gürültüsüz versiyonu çağırır (varsayılan olarak)
    }
    generated_data = []
    print("Gürültüsüz (ama değişken aralıklı) formasyon verileri üretiliyor...")
    for formation_name, generator_func in formation_generators.items():
        print(f"  -> {samples_per_formation} adet '{formation_name}' üretiliyor...")
        for i in range(samples_per_formation):
            # Fonksiyonu varsayılan (gürültüsüz) parametrelerle çağır
            sample = generator_func()
            generated_data.append(sample)

    print(f"\nÜretim tamamlandı. Toplam {len(generated_data)} örnek üretildi.")
    if generated_data:
        print("\nÜretilen rastgele bir örnek:")
        print(json.dumps(random.choice(generated_data), indent=2))

    output_filename = "generated_staggered_column_noiseless_variable_spacing.json"
    try:
        with open(output_filename, 'w') as f:
            json.dump(generated_data, f, indent=2)
        print(f"\nÜretilen veri '{output_filename}' dosyasına kaydedildi.")
    except Exception as e:
        print(f"\nDosyaya yazma hatası: {e}")