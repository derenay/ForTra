import random
import math
import json

# -----------------------------------------
# Düzenli (no noise) formasyon generatorleri
# -----------------------------------------
def generate_regular_line(num_objects, center, spread):
    cx, cy = center
    coords = []
    for i in range(num_objects):
        offset = spread * (i - (num_objects - 1) / 2)
        x = cx + offset
        y = cy
        coords.append([x, y])
    return coords

def generate_regular_wedge(num_objects, center, spread):
    # Lider tank merkeze yerleştirilir; sonra soldan ve sağdan eklenir.
    cx, cy = center
    coords = []
    coords.append([cx, cy])
    i = 1
    while len(coords) < num_objects:
        left_x = cx - spread * i
        left_y = cy - spread * i
        coords.append([left_x, left_y])
        if len(coords) >= num_objects:
            break
        right_x = cx + spread * i
        right_y = cy - spread * i
        coords.append([right_x, right_y])
        i += 1
    return coords[:num_objects]

def generate_regular_vee(num_objects, center, spread):
    # Vee formasyonu: Ucu önde, iki kol yukarıya doğru açılır.
    cx, cy = center
    coords = []
    coords.append([cx, cy])
    i = 1
    while len(coords) < num_objects:
        left_x = cx - spread * i
        left_y = cy + spread * i
        coords.append([left_x, left_y])
        if len(coords) >= num_objects:
            break
        right_x = cx + spread * i
        right_y = cy + spread * i
        coords.append([right_x, right_y])
        i += 1
    return coords[:num_objects]

def generate_regular_echelon_right(num_objects, center, spread):
    # Tanklar sağa doğru kayarak yerleşir; yatay ve dikey offsetler kullanılır.
    cx, cy = center
    coords = []
    for i in range(num_objects):
        x = cx + spread * i
        y = cy - spread * i * 0.8
        coords.append([x, y])
    return coords

def generate_regular_herringbone(num_objects, center, spread):
    # Herringbone: Tanklar, çapraz (V veya X) desende yerleşir.
    cx, cy = center
    coords = []
    for i in range(num_objects):
        # Alternatif olarak sol ve sağa offset eklenir.
        offset = spread * 0.5 if i % 2 == 0 else -spread * 0.5
        x = cx + offset
        y = cy - spread * i * 0.8
        coords.append([x, y])
    return coords

def generate_regular_coil(num_objects, center, radius):
    # Spiral (coil) formasyon: Tanklar dairesel olarak yayılır.
    cx, cy = center
    coords = []
    total_angle = math.pi * 1.5
    for i in range(num_objects):
        angle = total_angle * (i / num_objects)
        r = radius * (i + 1) / num_objects
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        coords.append([x, y])
    return coords

def generate_regular_staggered_column(num_objects, center, spread):
    # Tanklar dikey sütun oluşturur; her satırda hafif yatay offset bulunur.
    cx, cy = center
    coords = []
    for i in range(num_objects):
        offset = spread * 0.3 if i % 2 == 0 else -spread * 0.3
        x = cx + offset
        y = cy - spread * i
        coords.append([x, y])
    return coords

def generate_regular_echelon_left(num_objects, center, spread):
    # Echelon Left: Tanklar sola doğru kayar.
    cx, cy = center
    coords = []
    for i in range(num_objects):
        x = cx - spread * i
        y = cy - spread * i * 0.8
        coords.append([x, y])
    return coords

def generate_regular_column(num_objects, center, spread):
    # Column: Tanklar tam dikey, aynı x, azalan y ile sıralanır.
    cx, cy = center
    coords = []
    for i in range(num_objects):
        x = cx
        y = cy - spread * i
        coords.append([x, y])
    return coords

def generate_regular_echelon(num_objects, center, spread):
    # Echelon: Lider tank merkeze, ardından soldan ve sağdan sabit offsetlerle yerleşir.
    cx, cy = center
    coords = []
    coords.append([cx, cy])
    i = 1
    while len(coords) < num_objects:
        left_x = cx - spread * i
        left_y = cy - spread * i * 1.2
        coords.append([left_x, left_y])
        if len(coords) >= num_objects:
            break
        right_x = cx + spread * i
        right_y = cy - spread * i * 1.2
        coords.append([right_x, right_y])
        i += 1
    return coords[:num_objects]

# -----------------------------------------
# Rastgele (noisy) formasyon generatorleri
# -----------------------------------------
def generate_line(num_objects, center, spread):
    cx, cy = center
    coords = []
    angle = random.uniform(-math.pi / 18, math.pi / 18)
    for i in range(num_objects):
        offset = spread * (i - (num_objects - 1) / 2)
        x = cx + offset * math.cos(angle) + random.uniform(-spread * 0.05, spread * 0.05)
        y = cy + offset * math.sin(angle) + random.uniform(-spread * 0.05, spread * 0.05)
        coords.append([x, y])
    return coords

def generate_wedge(num_objects, center, spread):
    cx, cy = center
    coords = []
    coords.append([cx + random.uniform(-spread * 0.05, spread * 0.05),
                   cy + random.uniform(-spread * 0.05, spread * 0.05)])
    i = 1
    while len(coords) < num_objects:
        left_x = cx - spread * i + random.uniform(-spread * 0.05, spread * 0.05)
        left_y = cy - spread * i + random.uniform(-spread * 0.05, spread * 0.05)
        coords.append([left_x, left_y])
        if len(coords) >= num_objects:
            break
        right_x = cx + spread * i + random.uniform(-spread * 0.05, spread * 0.05)
        right_y = cy - spread * i + random.uniform(-spread * 0.05, spread * 0.05)
        coords.append([right_x, right_y])
        i += 1
    return coords[:num_objects]

def generate_vee(num_objects, center, spread):
    cx, cy = center
    coords = []
    coords.append([cx + random.uniform(-spread * 0.05, spread * 0.05),
                   cy + random.uniform(-spread * 0.05, spread * 0.05)])
    i = 1
    while len(coords) < num_objects:
        left_x = cx - spread * i + random.uniform(-spread * 0.05, spread * 0.05)
        left_y = cy + spread * i + random.uniform(-spread * 0.05, spread * 0.05)
        coords.append([left_x, left_y])
        if len(coords) >= num_objects:
            break
        right_x = cx + spread * i + random.uniform(-spread * 0.05, spread * 0.05)
        right_y = cy + spread * i + random.uniform(-spread * 0.05, spread * 0.05)
        coords.append([right_x, right_y])
        i += 1
    return coords[:num_objects]

def generate_echelon_right(num_objects, center, spread):
    cx, cy = center
    coords = []
    step = spread * 0.8
    for i in range(num_objects):
        x = cx + step * i + random.uniform(-spread * 0.05, spread * 0.05)
        y = cy - step * i + random.uniform(-spread * 0.05, spread * 0.05)
        coords.append([x, y])
    return coords

def generate_herringbone(num_objects, center, spread):
    cx, cy = center
    coords = []
    for i in range(num_objects):
        offset = spread * 0.5 if i % 2 == 0 else -spread * 0.5
        x = cx + offset + random.uniform(-spread * 0.05, spread * 0.05)
        y = cy - spread * i * 0.8 + random.uniform(-spread * 0.05, spread * 0.05)
        coords.append([x, y])
    return coords

def generate_coil(num_objects, center, radius):
    cx, cy = center
    coords = []
    total_angle = math.pi * 1.5
    for i in range(num_objects):
        angle = total_angle * (i / num_objects)
        r = radius * (i + 1) / num_objects
        x = cx + r * math.cos(angle) + random.uniform(-radius * 0.05, radius * 0.05)
        y = cy + r * math.sin(angle) + random.uniform(-radius * 0.05, radius * 0.05)
        coords.append([x, y])
    return coords

def generate_staggered_column(num_objects, center, spread):
    cx, cy = center
    coords = []
    for i in range(num_objects):
        offset = spread * 0.3 if i % 2 == 0 else -spread * 0.3
        x = cx + offset + random.uniform(-spread * 0.05, spread * 0.05)
        y = cy - spread * i + random.uniform(-spread * 0.05, spread * 0.05)
        coords.append([x, y])
    return coords

def generate_echelon_left(num_objects, center, spread):
    cx, cy = center
    coords = []
    step = spread * 0.8
    for i in range(num_objects):
        x = cx - step * i + random.uniform(-spread * 0.05, spread * 0.05)
        y = cy - step * i + random.uniform(-spread * 0.05, spread * 0.05)
        coords.append([x, y])
    return coords

def generate_column(num_objects, center, spread):
    cx, cy = center
    coords = []
    for i in range(num_objects):
        x = cx + random.uniform(-spread * 0.05, spread * 0.05)
        y = cy - spread * i + random.uniform(-spread * 0.05, spread * 0.05)
        coords.append([x, y])
    return coords

def generate_echelon(num_objects, center, spread):
    cx, cy = center
    coords = []
    # Lider tank (gürültü eklenmiş)
    coords.append([cx + random.uniform(-spread * 0.05, spread * 0.05),
                   cy + random.uniform(-spread * 0.05, spread * 0.05)])
    row = 1
    while len(coords) < num_objects:
        left_x = cx - spread * row * random.uniform(0.9, 1.1)
        left_y = cy - spread * row * random.uniform(0.5, 0.7)
        coords.append([left_x + random.uniform(-spread * 0.05, spread * 0.05),
                       left_y + random.uniform(-spread * 0.05, spread * 0.05)])
        if len(coords) >= num_objects:
            break
        right_x = cx + spread * row * random.uniform(0.9, 1.1)
        right_y = cy - spread * row * random.uniform(0.5, 0.7)
        coords.append([right_x + random.uniform(-spread * 0.05, spread * 0.05),
                       right_y + random.uniform(-spread * 0.05, spread * 0.05)])
        row += 1
    return coords[:num_objects]

def generate_regular_echelon(num_objects, center, spread):
    cx, cy = center
    coords = []
    coords.append([cx, cy])
    row = 1
    while len(coords) < num_objects:
        left_x = cx - spread * row
        left_y = cy - spread * row * 1.2
        coords.append([left_x, left_y])
        if len(coords) >= num_objects:
            break
        right_x = cx + spread * row
        right_y = cy - spread * row * 1.2
        coords.append([right_x, right_y])
        row += 1
    return coords[:num_objects]

# -----------------------------------------
# Parametreler ve generator mapping'leri
# -----------------------------------------
formation_params = {
    "Line": {"num_objects": (4, 12), "spread": (0.05, 0.10)},
    "Wedge": {"num_objects": (4, 12), "spread": (0.05, 0.10)},
    "Vee": {"num_objects": (3, 12), "spread": (0.05, 0.10)},
    "Echelon Right": {"num_objects": (3, 10), "spread": (0.05, 0.10)},
    "Herringbone": {"num_objects": (4, 12), "spread": (0.05, 0.10)},
    "Coil": {"num_objects": (4, 15), "radius": (0.05, 0.10)},
    "Staggered Column": {"num_objects": (4, 10), "spread": (0.05, 0.10)},
    "Echelon Left": {"num_objects": (4, 10), "spread": (0.05, 0.10)},
    "Column": {"num_objects": (4, 10), "spread": (0.05, 0.10)},
    "Echelon": {"num_objects": (4, 10), "spread": (0.05, 0.10)}
}

formation_generators = {
    "Line": generate_line,
    "Wedge": generate_wedge,
    "Vee": generate_vee,
    "Echelon Right": generate_echelon_right,
    "Herringbone": generate_herringbone,
    "Coil": generate_coil,
    "Staggered Column": generate_staggered_column,
    "Echelon Left": generate_echelon_left,
    "Column": generate_column,
    "Echelon": generate_echelon
}

regular_generators = {
    "Line": generate_regular_line,
    "Wedge": generate_regular_wedge,
    "Vee": generate_regular_vee,
    "Echelon Right": generate_regular_echelon_right,
    "Herringbone": generate_regular_herringbone,
    "Coil": generate_regular_coil,
    "Staggered Column": generate_regular_staggered_column,
    "Echelon Left": generate_regular_echelon_left,
    "Column": generate_regular_column,
    "Echelon": generate_regular_echelon
}

# -----------------------------------------
# Veri Üretimi
# -----------------------------------------
synthetic_data = []
samples_per_formation = 1
formations = list(formation_generators.keys())

for formation in formations:
    noisy_generator = formation_generators[formation]
    regular_generator = regular_generators[formation]
    params = formation_params[formation]
    
    for _ in range(samples_per_formation):
        num_objects = random.randint(params["num_objects"][0], params["num_objects"][1])
        center = (round(random.uniform(0.35, 0.65), 2),
                  round(random.uniform(0.35, 0.65), 2))
        
        # %70 rastgele (noisy), %30 düzenli veri üretimi
        if random.random() < 0.7:
            if formation == "Coil":
                radius = round(random.uniform(params["radius"][0], params["radius"][1]), 2)
                coords = noisy_generator(num_objects, center, radius)
            else:
                spread = round(random.uniform(params["spread"][0], params["spread"][1]), 2)
                coords = noisy_generator(num_objects, center, spread)
        else:
            if formation == "Coil":
                radius = round(random.uniform(params["radius"][0], params["radius"][1]), 2)
                coords = regular_generator(num_objects, center, radius)
            else:
                spread = round(random.uniform(params["spread"][0], params["spread"][1]), 2)
                coords = regular_generator(num_objects, center, spread)
        
        # Her örnek için rastgele bir base angle belirleyip,
        # her tank için küçük gürültü eklenmiş yön hesaplanır.
        base_angle = random.uniform(0, 2 * math.pi)
        angles = []
        for _ in range(num_objects):
            noise_angle = random.uniform(-0.1, 0.1)
            tank_angle = base_angle + noise_angle
            tank_angle_norm = (tank_angle % (2 * math.pi)) / (2 * math.pi)
            angles.append(round(tank_angle_norm, 3))
        
        sample = {
            "coordinates": [[round(x, 3) for x in point] for point in coords],
            "classes": ["tank"] * len(coords),
            "formation": formation,
            "angles": angles
        }
        synthetic_data.append(sample)

# JSON olarak kaydet
with open("tank_formations_mixed.json", "w") as f:
    json.dump(synthetic_data, f, indent=2)

print(f"Veri seti oluşturuldu: {len(synthetic_data)} örnek")
