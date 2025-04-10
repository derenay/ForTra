import random
import math
import json
import numpy as np
# -----------------------------------------
# Yardımcı Fonksiyonlar
# -----------------------------------------
def rotate_point(point, center, angle):
    """Bir noktayı belirli bir merkez etrafında döndürür."""
    x, y = point
    cx, cy = center
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    nx = cx + (x - cx) * cos_a - (y - cy) * sin_a
    ny = cy + (x - cx) * sin_a + (y - cy) * cos_a
    return nx, ny

def normalize_angle(angle):
    """Açıyı [0, 2*pi) aralığına getirir."""
    return angle % (2 * math.pi)

def angle_to_normalized(angle):
    """Açıyı [0, 2*pi) aralığından [0, 1) aralığına normalize eder."""
    return normalize_angle(angle) / (2 * math.pi)

# -----------------------------------------
# Düzenli (noise'suz) formasyon generatorleri (Varsayılan Yönelimde)
# -----------------------------------------
# Not: Bu fonksiyonlar formasyonu genellikle Y ekseni negatif yönde (aşağı) veya
#      X ekseni boyunca oluşturur. Döndürme işlemi ana döngüde yapılır.


def generate_column(
    num_units_range=(4, 12),
    axis_direction_norm=None,
    unit_sep_mean=0.1,           # Birimler arası sabit mesafe olacak
    # unit_sep_std=0.0,         # <<< GÜRÜLTÜSÜZ: Standart sapma kullanılmıyor veya 0
    coord_noise_scale=0.0,      # <<< GÜRÜLTÜSÜZ: Koordinat gürültüsü varsayılanı 0
    direction_noise_scale=0.01,  # <<< GÜRÜLTÜSÜZ: Yön gürültüsü varsayılanı 0
    center_x_range=(0.1, 0.9),
    center_y_range=(0.1, 0.9),
    # --- İsteğe Bağlı Kesin Parametreler ---
    exact_num_units=None,
    exact_start_pos=None,
    exact_axis_direction_norm=None,
    exact_unit_sep=None
) -> dict:
    """
    Belirtilen parametrelere göre (varsayılan olarak GÜRÜLTÜSÜZ) bir Column
    formasyon örneği oluşturur. Gürültü parametreleri 0 ise deterministik (ideal) olur.

    Returns:
        dict: 'coordinates', 'classes', 'formation', 'directions' anahtarlarını
              içeren bir dictionary.
    """

    # Birim sayısı
    if exact_num_units is not None:
        num_units = exact_num_units
    else:
        # Eğer gürültüsüz ve hep aynı sayıda isteniyorsa range yerine sabit değer verilebilir
        num_units = random.randint(num_units_range[0], num_units_range[1])
    if num_units < 1: num_units = 1

    # Başlangıç pozisyonu
    if exact_start_pos is not None:
        start_x, start_y = exact_start_pos
    else:
        start_x = random.uniform(center_x_range[0], center_x_range[1])
        start_y = random.uniform(center_y_range[0], center_y_range[1])

    # Eksen yönü
    if exact_axis_direction_norm is not None:
        _axis_direction_norm = exact_axis_direction_norm
    elif axis_direction_norm is not None:
         _axis_direction_norm = axis_direction_norm
    else:
        _axis_direction_norm = random.random() # Hala rastgele yön seçilebilir
    axis_angle_rad = _axis_direction_norm * 2 * math.pi

    # Birim ayrımı (Artık standart sapma yok)
    unit_sep = exact_unit_sep if exact_unit_sep is not None else unit_sep_mean
    if unit_sep <= 0: unit_sep = 0.1 # Negatif/sıfır olmasın

    coordinates = []
    directions_rad = []
    axis_dx = math.cos(axis_angle_rad)
    axis_dy = math.sin(axis_angle_rad)
    current_pos_x = start_x
    current_pos_y = start_y

    for i in range(num_units):
        unit_x = current_pos_x
        unit_y = current_pos_y

        # Koordinatlara gürültü ekle (coord_noise_scale 0 ise 0 eklenir)
        # np.random.normal(0, 0.0) -> 0.0 döndürür
        noise_x = np.random.normal(0, coord_noise_scale)
        noise_y = np.random.normal(0, coord_noise_scale)
        final_x = unit_x + noise_x
        final_y = unit_y + noise_y

        # YÖN HESAPLAMA: Ana eksen yönü + gürültü (direction_noise_scale 0 ise 0 eklenir)
        direction_noise_rad = np.random.normal(0, direction_noise_scale * 2 * math.pi)
        final_direction_rad = axis_angle_rad + direction_noise_rad

        coordinates.append([final_x, final_y])
        directions_rad.append(final_direction_rad)

        # Bir sonraki birimin ideal konumu (sabit 'unit_sep' mesafesiyle)
        current_pos_x += axis_dx * unit_sep
        current_pos_y += axis_dy * unit_sep

    # Radyan yönleri 0-1 arasına normalize et
    directions_norm = [(angle / (2 * math.pi)) % 1.0 for angle in directions_rad]

    result = {
        "coordinates": coordinates,
        "classes": ["tank"] * num_units,
        "formation": "Column",
        "directions": directions_norm
    }
    return result


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


def generate_herringbone(
    num_units_range=(4, 12),       # Formasyondaki min/max birim sayısı
    axis_direction_norm=None,    # Formasyonun ana yönü (0-1 arası, None ise rastgele)
    unit_sep_mean=0.08,          # Eksen boyunca birimler arası ortalama mesafe
    unit_sep_std=0.02,           # Mesafe standard sapması (çeşitlilik için)
    offset_dist_mean=0.05,       # Ekseninden sağa/sola kayma mesafesi (ortalama)
    offset_dist_std=0.01,        # Kayma mesafesi standard sapması
    angle_offset_norm_mean=0.1,  # Birimlerin eksenden dışa doğru sapma açısı (0-1 arası, ortalama)
                                 # 0.1 -> 0.1*360 = 36 derece sapma gibi
    angle_offset_norm_std=0.03,  # Açı sapması standard sapması
    coord_noise_scale=0.008,     # Koordinatlara eklenecek gürültü ölçeği
    direction_noise_scale=0.01,  # Yönlere eklenecek gürültü ölçeği
    center_x_range=(0.2, 0.8),   # Formasyon merkezinin X koordinat aralığı
    center_y_range=(0.2, 0.8)    # Formasyon merkezinin Y koordinat aralığı
) -> dict:
    """
    Belirtilen parametrelere göre rastgele bir Herringbone formasyon örneği oluşturur.

    Returns:
        dict: 'coordinates', 'classes', 'formation', 'directions' anahtarlarını
              içeren bir dictionary.
    """

    # Rastgele birim sayısı belirle
    num_units = random.randint(num_units_range[0], num_units_range[1])
    if num_units < 2: num_units = 2 # En az 2 birim

    # Rastgele merkez nokta belirle
    center_x = random.uniform(center_x_range[0], center_x_range[1])
    center_y = random.uniform(center_y_range[0], center_y_range[1])

    # Rastgele ana eksen yönü belirle (eğer verilmediyse)
    if axis_direction_norm is None:
        axis_direction_norm = random.random()
    axis_angle_rad = axis_direction_norm * 2 * math.pi

    # Açı sapmasını belirle (radyan cinsinden)
    angle_offset_norm = np.random.normal(angle_offset_norm_mean, angle_offset_norm_std)
    # Sapmanın 0'dan büyük olmasını sağla (negatif olmasın)
    angle_offset_norm = max(0.01, angle_offset_norm)
    angle_offset_rad = angle_offset_norm * 2 * math.pi

    coordinates = []
    directions_rad = [] # Yönleri önce radyan olarak hesaplayalım

    # Eksen yön vektörü (normalize edilmiş)
    axis_dx = math.cos(axis_angle_rad)
    axis_dy = math.sin(axis_angle_rad)
    # Eksene dik yön vektörü (sola doğru offset için)
    perp_dx = -axis_dy
    perp_dy = axis_dx

    current_pos_x = center_x
    current_pos_y = center_y

    for i in range(num_units):
        # Birimler arası mesafeyi belirle
        sep = np.random.normal(unit_sep_mean, unit_sep_std)
        sep = max(0.01, sep) # Negatif veya sıfır olmasın

        # Eksen üzerindeki ideal konumu ilerlet
        if i > 0: # İlk birim merkezde başlar
            current_pos_x += axis_dx * sep
            current_pos_y += axis_dy * sep

        # Sağa/sola kayma mesafesini belirle
        offset_dist = np.random.normal(offset_dist_mean, offset_dist_std)
        offset_dist = max(0.01, offset_dist)

        # Sağa veya sola kaydır
        unit_x = current_pos_x
        unit_y = current_pos_y
        direction_rad = axis_angle_rad # Başlangıç yönü eksen yönü

        if i % 2 == 1: # Tek indeksliler (1, 3, 5...) sağa kaysın ve sağa dönsün
            unit_x -= perp_dx * offset_dist # Sağa kaydırma (perp vektörü sola doğruydu)
            unit_y -= perp_dy * offset_dist
            direction_rad -= angle_offset_rad # Sağa doğru açı sapması
        else: # Çift indeksliler (0, 2, 4...) sola kaysın ve sola dönsün
            unit_x += perp_dx * offset_dist # Sola kaydırma
            unit_y += perp_dy * offset_dist
            direction_rad += angle_offset_rad # Sola doğru açı sapması

        # Koordinatlara gürültü ekle
        noise_x = np.random.normal(0, coord_noise_scale)
        noise_y = np.random.normal(0, coord_noise_scale)
        final_x = unit_x + noise_x
        final_y = unit_y + noise_y

        # Yöne gürültü ekle (radyan cinsinden)
        direction_noise_rad = np.random.normal(0, direction_noise_scale * 2 * math.pi)
        final_direction_rad = direction_rad + direction_noise_rad

        coordinates.append([final_x, final_y])
        directions_rad.append(final_direction_rad)

    # Radyan yönleri 0-1 arasına normalize et
    directions_norm = [(angle / (2 * math.pi)) % 1.0 for angle in directions_rad]

    # Sonucu dictionary olarak formatla
    result = {
        "coordinates": coordinates,
        "classes": ["tank"] * num_units, # Şimdilik hepsi tank
        "formation": "Herringbone",
        "directions": directions_norm
    }
    return result


def generate_regular_line(num_objects, center, spread):
    # Yatay çizgi oluşturur (hareket yönü varsayılan olarak yukarı/aşağı)
    cx, cy = center
    coords = []
    for i in range(num_objects):
        offset = spread * (i - (num_objects - 1) / 2)
        x = cx + offset
        y = cy # Başlangıçta y sabit
        coords.append([x, y])
    # Hareket yönünü belirlemek için ilk ve son noktayı kullanabiliriz (döndürme sonrası)
    # Açıları ana döngüde belirle
    return coords

def generate_regular_wedge(num_objects, center, spread):
    # Kama formasyonu (varsayılan olarak aşağı doğru sivrilir)
    cx, cy = center
    coords = []
    coords.append([cx, cy])  # Lider tank merkezde (en önde/yukarıda)
    i = 1
    while len(coords) < num_objects:
        # Simetrik olarak sola ve sağa, geriye doğru ekle
        left_x = cx - spread * i * 0.7 # Daha dar kama için katsayı
        left_y = cy - spread * i
        coords.append([left_x, left_y])
        if len(coords) >= num_objects: break
        right_x = cx + spread * i * 0.7
        right_y = cy - spread * i
        coords.append([right_x, right_y])
        i += 1
    return coords[:num_objects]

def generate_regular_vee(num_objects, center, spread):
    # V formasyonu (varsayılan olarak yukarı doğru açılır)
    cx, cy = center
    coords = []
    coords.append([cx, cy])  # Uç noktası (en geride/aşağıda)
    i = 1
    while len(coords) < num_objects:
        # Simetrik olarak sola ve sağa, ileriye doğru ekle
        left_x = cx - spread * i * 0.7
        left_y = cy + spread * i
        coords.append([left_x, left_y])
        if len(coords) >= num_objects: break
        right_x = cx + spread * i * 0.7
        right_y = cy + spread * i
        coords.append([right_x, right_y])
        i += 1
    return coords[:num_objects]



def generate_regular_coil(num_objects, center, radius_factor):
    # Sarmal (Coil/Leaguer) Formasyonu (Durma/Savunma)
    cx, cy = center
    coords = []
    # Daha çok dairesel bir yerleşim yapar, tam sarmal olmayabilir
    angle_step = 2 * math.pi / num_objects
    base_radius = radius_factor * num_objects * 0.3 # Obje sayısına göre yarıçap

    for i in range(num_objects):
        angle = i * angle_step
        # Yarıçapı hafifçe rastgele veya sıralı değiştirebiliriz
        r = base_radius * (1 + (i%2)*0.1) # Hafif iç/dış yapabilir
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        coords.append([x, y])
    # Açıları ana döngüde belirlenecek (merkezden dışa dönük)
    return coords


def generate_regular_echelon(num_objects, center, spread):
    # Eşelon (Genel) - Wedge'e benzer ama lider merkezde olmayabilir, sağa/sola kayık başlar
    # Bu implementasyon Wedge'e benziyor, lider önde merkezde.
    # Farklı bir yorum: Lider bir tarafta, diğerleri çapraz geride. Echelon Right/Left daha spesifik.
    # Şimdilik Wedge ile aynı mantıkta bırakalım, gerekirse Echelon Right/Left kullanılır.
    return generate_regular_wedge(num_objects, center, spread)


# -----------------------------------------
# Rastgele (noisy) formasyon generatorleri
# -----------------------------------------
# Not: Bunlar da varsayılan yönelimde üretir, sonra döndürülür.
#      Noise, hem pozisyona hem de elemanlar arası mesafeye eklenebilir.

def generate_line(num_objects, center, spread):
    coords = generate_regular_line(num_objects, center, spread)
    noisy_coords = []
    noise_factor = spread * 0.15 # Gürültü miktarını spread ile oranla
    for x, y in coords:
        nx = x + random.uniform(-noise_factor, noise_factor)
        ny = y + random.uniform(-noise_factor, noise_factor)
        noisy_coords.append([nx, ny])
    return noisy_coords

def generate_wedge(num_objects, center, spread):
    coords = generate_regular_wedge(num_objects, center, spread)
    noisy_coords = []
    noise_factor = spread * 0.15
    for x, y in coords:
        nx = x + random.uniform(-noise_factor, noise_factor)
        ny = y + random.uniform(-noise_factor, noise_factor)
        noisy_coords.append([nx, ny])
    return noisy_coords

def generate_vee(num_objects, center, spread):
    coords = generate_regular_vee(num_objects, center, spread)
    noisy_coords = []
    noise_factor = spread * 0.15
    for x, y in coords:
        nx = x + random.uniform(-noise_factor, noise_factor)
        ny = y + random.uniform(-noise_factor, noise_factor)
        noisy_coords.append([nx, ny])
    return noisy_coords




def generate_coil(num_objects, center, radius_factor):
     # Regular coil zaten biraz dağınık, noise ekleyelim
    coords = generate_regular_coil(num_objects, center, radius_factor)
    noisy_coords = []
    # Coil için noise'u radius_factor'a göre ayarlayalım
    noise_factor = radius_factor * num_objects * 0.05
    for x, y in coords:
        nx = x + random.uniform(-noise_factor, noise_factor)
        ny = y + random.uniform(-noise_factor, noise_factor)
        noisy_coords.append([nx, ny])
    return noisy_coords



def generate_echelon(num_objects, center, spread):
    # Regular Echelon (Wedge gibi) üzerine noise ekleyelim
    coords = generate_regular_echelon(num_objects, center, spread)
    noisy_coords = []
    noise_factor = spread * 0.15
    for x, y in coords:
        nx = x + random.uniform(-noise_factor, noise_factor)
        ny = y + random.uniform(-noise_factor, noise_factor)
        noisy_coords.append([nx, ny])
    return noisy_coords

# -----------------------------------------
# Parametreler ve generator mapping'leri
# -----------------------------------------
formation_params = {
    "Line": {"num_objects": (3, 12), "spread": (0.05, 0.10)},
    "Wedge": {"num_objects": (3, 12), "spread": (0.05, 0.10)},
    "Vee": {"num_objects": (3, 12), "spread": (0.05, 0.10)},
    "Coil": {"num_objects": (3, 15), "radius_factor": (0.04, 0.08)}, # radius_factor birim başına yarıçapı etkiler
    "Echelon": {"num_objects": (3, 10), "spread": (0.05, 0.10)} # Wedge gibi davranıyor
}

# Noisy ve Regular generatorları ayrı ayrı tanımla
formation_generators = {
    "Line": generate_line,
    "Wedge": generate_wedge,
    "Vee": generate_vee,
    "Coil": generate_coil,
    "Echelon": generate_echelon 
}

regular_generators = {
    "Line": generate_regular_line,
    "Wedge": generate_regular_wedge,
    "Vee": generate_regular_vee,
    "Coil": generate_regular_coil,
    "Echelon": generate_regular_echelon 
}

formation_generators_new = {
        'Herringbone': generate_herringbone,
        'Staggered Column': generate_staggered_column,
        'Column': generate_column
    }


synthetic_data = []
samples_per_formation = 20
formations = list(formation_generators.keys())
noise_ratio = 0.3

for formation in formations:
    noisy_generator = formation_generators[formation]
    regular_generator = regular_generators[formation]
    params = formation_params[formation]

    for i in range(samples_per_formation):
        num_objects = random.randint(params["num_objects"][0], params["num_objects"][1])
        # Merkezi 0.1-0.9 arasında rastgele seçelim ki kenarlara taşmasın
        center = (round(random.uniform(0.15, 0.85), 2),
                  round(random.uniform(0.15, 0.85), 2))

        is_noisy = random.random() < noise_ratio
        generator = noisy_generator if is_noisy else regular_generator

        # Formasyon özel parametrelerini al
        if formation == "Coil":
            radius_factor = round(random.uniform(params["radius_factor"][0], params["radius_factor"][1]), 3)
            initial_coords = generator(num_objects, (0,0), radius_factor) # Önce (0,0) etrafında üret
        else:
            spread_val = round(random.uniform(params["spread"][0], params["spread"][1]), 3)
            initial_coords = generator(num_objects, (0,0), spread_val) # Önce (0,0) etrafında üret

        # --- Döndürme ve Kaydırma ---
        rotation_angle = random.uniform(0, 2 * math.pi) # Rastgele döndürme açısı
        final_coords = []
        for point in initial_coords:
            # Önce (0,0) etrafında döndür
            rotated_point = rotate_point(point, (0,0), rotation_angle)
            # Sonra hedef merkeze kaydır
            final_point = (rotated_point[0] + center[0], rotated_point[1] + center[1])
            final_coords.append(list(final_point))


          # --- Yön (Angle) Belirleme ---
        angles = []
        angle_noise = 0.1 # Açıya eklenecek maksimum gürültü (radyan)
        formation_direction = rotation_angle # Formasyonun genel döndürme açısı

        # Formasyonların varsayılan üretim yönelimleri (döndürme öncesi):
        # - Line: X ekseni boyunca yatay. Hareket genelde Y ekseninde (+/- pi/2). Tanklar hatta dik bakar.
        
        # - Wedge/Echelon: Y ekseni boyunca, sivri ucu aşağı (-Y) doğru. Varsayılan Hareket Yönü = -pi/2 radyan. Tanklar bu yöne bakar.
        # - Vee: Y ekseni boyunca, açık ucu yukarı (+Y) doğru. Varsayılan Hareket Yönü = +pi/2 radyan. Tanklar bu yöne bakar.
        # - Coil: Merkez etrafında dairesel durma formasyonu. Tanklar dışa bakar.

        if formation == "Line":
            # Tanklar çizgiye dik bakar. Çizginin varsayılan yönelimi 0 radyan (X ekseni).
            # Tankların varsayılan bakış yönü: 0 + pi/2 = +pi/2 (Yukarı).
            # Son yön = rotation_angle + pi/2
            base_angle = formation_direction + math.pi / 2
            for _ in range(num_objects):
                 noise = random.uniform(-angle_noise, angle_noise)
                 angles.append(angle_to_normalized(base_angle + noise))

                
        elif formation == ["Column"]:
            # Varsayılan hareket yönü aşağı (-pi/2). Tanklar bu yöne bakar.
            # Son yön = rotation_angle + (-pi/2)
            base_angle = formation_direction - math.pi / 2
            for _ in range(num_objects):
                noise = random.uniform(-angle_noise, angle_noise)
                angles.append(angle_to_normalized(base_angle + noise))


        elif formation == "Wedge":
            # Varsayılan hareket yönü (sivri ucun gösterdiği yön) aşağı (-pi/2). Tanklar bu yöne bakar.
            # Son yön = rotation_angle + (-pi/2)
            base_angle = formation_direction + math.pi / 2
            for _ in range(num_objects):
                noise = random.uniform(-angle_noise * 1.5, angle_noise * 1.5) # Echelon/Wedge'de hafif sapma fazla olabilir
                angles.append(angle_to_normalized(base_angle + noise))
                 
        elif formation == "Echelon":
             # Varsayılan hareket yönü (sivri ucun gösterdiği yön) aşağı (-pi/2). Tanklar bu yöne bakar.
             # Son yön = rotation_angle + (-pi/2)
             base_angle = formation_direction + math.pi / 2
             for _ in range(num_objects):
                 noise = random.uniform(-angle_noise * 1.5, angle_noise * 1.5) # Echelon/Wedge'de hafif sapma fazla olabilir
                 angles.append(angle_to_normalized(base_angle + noise))
        

        elif formation == "Vee":
             # ***** DÜZELTME BURADA *****
             # Varsayılan hareket yönü (açık ucun gösterdiği yön) yukarı (+pi/2). Tanklar bu yöne bakar.
             # Son yön = rotation_angle + (+pi/2)
             base_angle = formation_direction + math.pi / 2
             for _ in range(num_objects):
                 noise = random.uniform(-angle_noise * 1.5, angle_noise * 1.5) # Vee'de de hafif sapma fazla olabilir
                 angles.append(angle_to_normalized(base_angle + noise))



        elif formation == "Coil":
             # Coil Formasyonu: FM tanımı olmadığı için merkezden dışa bakış mantığı kalabilir.
             # Bu formasyon genellikle daha statik, dairesel bir savunma içindir.
             cx, cy = center
             for k, point_final in enumerate(final_coords):
                 x, y = point_final
                 if abs(x - cx) < 1e-9 and abs(y - cy) < 1e-9:
                     angle_from_center = math.pi / 2 # Merkezdeyse yukarı bak
                 else:
                     angle_from_center = math.atan2(y - cy, x - cx)
                 noise = random.uniform(-angle_noise*1.5, angle_noise*1.5)
                 angles.append(angle_to_normalized(angle_from_center + noise))


        elif formation == "Coil":
                # Bu zaten doğruydu - Tanklar merkezden dışa bakar.
                cx, cy = center
                for k, point_final in enumerate(final_coords):
                    x, y = point_final
                    # Merkezde olma durumunu kontrol etmek iyi olabilir
                    if abs(x - cx) < 1e-9 and abs(y - cy) < 1e-9:
                        angle_from_center = math.pi / 2 # Merkezdeyse yukarı bak
                    else:
                        angle_from_center = math.atan2(y - cy, x - cx)
                    noise = random.uniform(-angle_noise*1.5, angle_noise*1.5) # Coil'de gürültü biraz daha fazla olabilir
                    angles.append(angle_to_normalized(angle_from_center + noise))


        else: # Diğer/bilinmeyen durumlar için genel ileri yön (emin değilsek)
             # Varsayılan olarak Column gibi davransın (-pi/2 offset)
             base_angle = formation_direction - math.pi/2
             for _ in range(num_objects):
                 noise = random.uniform(-angle_noise, angle_noise)
                 angles.append(angle_to_normalized(base_angle + noise))

        # Koordinatları ve açıları yuvarla
        rounded_coords = [[round(x, 3) for x in point] for point in final_coords]
        rounded_angles = [round(a, 3) for a in angles]

        # Koordinatların [0, 1] aralığında kalmasını sağla (nadiren taşabilir)
        
        clipped_coords = []
        valid_sample = True
            
        
        for x, y in rounded_coords:
            # Koordinatları 0 ve 1 arasında kırp
            cx = max(0.0, min(1.0, x))
            cy = max(0.0, min(1.0, y))
            # Eğer kırpma yapıldıysa ve orijinal koordinat farklıysa, bu örnek geometriyi bozmuş olabilir.
            # Tolerans eklenebilir veya bu örnek atlanabilir. Şimdilik atlayalım.
            if abs(cx - x) > 1e-6 or abs(cy - y) > 1e-6: # Küçük floating point hatalarını göz ardı et
                # print(f"Uyarı: {formation} örneği sınırlara taştı ({x},{y} -> {cx},{cy}), atlanıyor.")
                valid_sample = False
                break
            clipped_coords.append([cx, cy])

        if not valid_sample:
            continue # Bu örneği atla

        sample = {
            "coordinates": clipped_coords,
            "classes": ["tank"] * len(clipped_coords),
            "formation": formation,
            "directions": rounded_angles, # [0, 1) aralığında normalize edilmiş açılar
        }
        synthetic_data.append(sample)
        
generated_data = []

for formation_name, generator_func in formation_generators_new.items():
    for i in range(samples_per_formation):
        sample = generator_func()
        generated_data.append(sample)
       
    


synthetic_data = synthetic_data + generated_data

# JSON olarak kaydet
output_filename = "tank_formations_mixed.json"
with open(output_filename, "w") as f:
    json.dump(synthetic_data, f, indent=2)

print(f"Veri seti oluşturuldu: {len(synthetic_data)} örnek, kaydedildi: {output_filename}")