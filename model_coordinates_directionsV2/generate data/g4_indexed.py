import random
import math
import json

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




def generate_regular_herringbone(num_objects, center, spread):
    # Kılçık (Balıksırtı) Formasyonu (Durma) - Araçlar merkezden dışa bakar
    # Genellikle bir eksen boyunca dizilirler, örneğin Y ekseni
    cx, cy = center
    coords = []
    coords.append([cx, cy]) # Lider
    left_count = (num_objects -1) // 2
    right_count = num_objects - 1 - left_count

    # Sol taraf (hafif sol-geri)
    for i in range(1, left_count + 1):
        x = cx - spread * 0.5 # Sabit x offset sola
        y = cy - spread * i   # Geriye doğru artan mesafe
        coords.append([x,y])

    # Sağ taraf (hafif sağ-geri)
    for i in range(1, right_count + 1):
        x = cx + spread * 0.5 # Sabit x offset sağa
        y = cy - spread * i   # Geriye doğru artan mesafe
        coords.append([x,y])

    # Not: Açıları ana döngüde belirlenecek (dışa dönük)
    return coords

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

def generate_regular_staggered_column(num_objects, center, spread):
    # Kademeli Kol (Staggered Column)
    cx, cy = center
    coords = []
    offset_amount = spread * 0.5 # Kademelenme miktarı
    for i in range(num_objects):
        x = cx + offset_amount * (i % 2) # Sırayla sağa/sola veya sadece sağa offset
        y = cy - spread * i # Geriye doğru (aşağı)
        coords.append([x, y])
    return coords

def generate_regular_column(num_objects, center, spread):
    # Kol (Column) Formasyonu
    cx, cy = center
    coords = []
    for i in range(num_objects):
        x = cx # Sabit X
        y = cy - spread * i # Geriye doğru (aşağı)
        coords.append([x, y])
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



def generate_herringbone(num_objects, center, spread):
    coords = generate_regular_herringbone(num_objects, center, spread)
    noisy_coords = []
    noise_factor = spread * 0.15
    # Liderin pozisyonunu koru (opsiyonel) veya ona da noise ekle
    for i, (x, y) in enumerate(coords):
        # if i == 0: # Lideri sabit tutmak istersen
        #     noisy_coords.append([x,y])
        #     continue
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


def generate_staggered_column(num_objects, center, spread):
    coords = generate_regular_staggered_column(num_objects, center, spread)
    noisy_coords = []
    noise_factor = spread * 0.15
    for x, y in coords:
        nx = x + random.uniform(-noise_factor, noise_factor)
        ny = y + random.uniform(-noise_factor, noise_factor)
        noisy_coords.append([nx, ny])
    return noisy_coords

def generate_column(num_objects, center, spread):
    coords = generate_regular_column(num_objects, center, spread)
    noisy_coords = []
    noise_factor = spread * 0.15
    for x, y in coords:
        # Kol formasyonunda Y eksenindeki sapma daha kritik olabilir
        nx = x + random.uniform(-noise_factor * 0.5, noise_factor * 0.5) # X'te daha az noise
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
    "Line": {"num_objects": (4, 12), "spread": (0.05, 0.10)},
    "Wedge": {"num_objects": (4, 12), "spread": (0.05, 0.10)},
    "Vee": {"num_objects": (3, 12), "spread": (0.05, 0.10)},
    "Echelon Right": {"num_objects": (3, 10), "spread": (0.05, 0.10)},
    "Echelon Left": {"num_objects": (3, 10), "spread": (0.05, 0.10)},
    "Herringbone": {"num_objects": (4, 12), "spread": (0.05, 0.10)}, # Spread burada eleman aralığı
    "Coil": {"num_objects": (4, 15), "radius_factor": (0.04, 0.08)}, # radius_factor birim başına yarıçapı etkiler
    "Staggered Column": {"num_objects": (4, 10), "spread": (0.05, 0.10)}, # Spread eleman aralığı
    "Column": {"num_objects": (4, 10), "spread": (0.05, 0.10)}, # Spread eleman aralığı
    "Echelon": {"num_objects": (4, 10), "spread": (0.05, 0.10)} # Wedge gibi davranıyor
}

# Noisy ve Regular generatorları ayrı ayrı tanımla
formation_generators = {
    "Line": generate_line,
    "Wedge": generate_wedge,
    "Vee": generate_vee,
    "Herringbone": generate_herringbone,
    "Coil": generate_coil,
    "Staggered Column": generate_staggered_column,
    "Column": generate_column,
    "Echelon": generate_echelon # Wedge'in noisy hali
}

regular_generators = {
    "Line": generate_regular_line,
    "Wedge": generate_regular_wedge,
    "Vee": generate_regular_vee,
    "Herringbone": generate_regular_herringbone,
    "Coil": generate_regular_coil,
    "Staggered Column": generate_regular_staggered_column,
    "Column": generate_regular_column,
    "Echelon": generate_regular_echelon # Wedge'in regular hali
}

# -----------------------------------------
# Veri Üretimi
# -----------------------------------------
synthetic_data = []
samples_per_formation = 6 # Örnek sayısını artırabiliriz
formations = list(formation_generators.keys())
noise_ratio = 0.7 # %70 noisy, %30 regular

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
        # - Column/Staggered: Y ekseni boyunca dikey, aşağı (-Y) doğru. Varsayılan Hareket Yönü = -pi/2 radyan. Tanklar bu yöne bakar.
        # - Wedge/Echelon: Y ekseni boyunca, sivri ucu aşağı (-Y) doğru. Varsayılan Hareket Yönü = -pi/2 radyan. Tanklar bu yöne bakar.
        # - Vee: Y ekseni boyunca, açık ucu yukarı (+Y) doğru. Varsayılan Hareket Yönü = +pi/2 radyan. Tanklar bu yöne bakar.
        # - Herringbone: Y ekseni boyunca aşağı (-Y) durma formasyonu. Eksen yönü -pi/2 radyan. Tanklar dışa bakar.
        # - Coil: Merkez etrafında dairesel durma formasyonu. Tanklar dışa bakar.

        if formation == "Line":
            # Tanklar çizgiye dik bakar. Çizginin varsayılan yönelimi 0 radyan (X ekseni).
            # Tankların varsayılan bakış yönü: 0 + pi/2 = +pi/2 (Yukarı).
            # Son yön = rotation_angle + pi/2
            base_angle = formation_direction + math.pi / 2
            for _ in range(num_objects):
                 noise = random.uniform(-angle_noise, angle_noise)
                 angles.append(angle_to_normalized(base_angle + noise))

        elif formation in ["Column", "Staggered Column"]:
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

        elif formation == "Herringbone":
            # ***** YENİDEN DÜZENLEME (FM Tanımına Göre) *****
            # Amaç: Hızlı savunma, 360 güvenlik, HAREKETE DEVAM ETME POZİSYONU,
            #       tanklar HAREKET YÖNÜNE ~45 DERECE açılı durur.
            axis_direction = formation_direction + math.pi / 2

            num_tanks = len(final_coords)
            cx = sum(x for x, y in final_coords) / num_tanks  # Merkez X koordinatı
            cy = sum(y for x, y in final_coords) / num_tanks  # Merkez Y koordinatı
        
            
            # 3. Hareket yönüne göre 45 derece (pi/4) sapma ile açıları ata.
            offset_angle = math.pi / 4  # 45 derece
          
            for k, point_orig in enumerate(final_coords):
                x, y = point_orig
                
                # Merkez noktasına olan çizgiyi hesapla
                if abs(x - cx) < 1e-9 and abs(y - cy) < 1e-9:
                    # Merkezdeki tank (lider): Merkezden dışarıya bakmalı.
                    tank_base_angle = axis_direction  # Hareket yönüne bakar, dışarıya yönlendirilmiş.
                else:
                    # Merkezden uzak tank: Merkez noktasına olan çizginin dışına doğru bakmalı.
                    dx = x - cx  # Merkez noktasına X uzaklığı
                    dy = y - cy  # Merkez noktasına Y uzaklığı
                    
                    # Merkez noktasına olan çizginin açısını hesapla
                    line_angle = math.atan2(dy, dx)
                    
                    # Çizginin tam tersini almak yerine, dışa doğru bakmak için çizgiyle aynı yönde bakmalı
                    tank_base_angle = line_angle  # Merkezden dışarıya doğru bakacak şekilde ayarla

                # Hesaplanan açıya küçük bir gürültü ekle
                noise = random.uniform(-angle_noise * 1.1, angle_noise * 1.1)  # Gürültü ile açıyı biraz rastgeleleştir
                tank_angle = tank_base_angle + noise
                angles.append(angle_to_normalized(tank_angle))




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
            "classes": 0 * len(clipped_coords),
            "formation": formation,
            "angles": rounded_angles, # [0, 1) aralığında normalize edilmiş açılar
        }
        synthetic_data.append(sample)

# JSON olarak kaydet
output_filename = "tank_formations_mixed.json"
with open(output_filename, "w") as f:
    json.dump(synthetic_data, f, indent=2)

print(f"Veri seti oluşturuldu: {len(synthetic_data)} örnek, kaydedildi: {output_filename}")