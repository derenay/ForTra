import random
import json
import math

def generate_line(num_objects=5, center=(0.5, 0.5), spread=0.1):
    cx, cy = center
    coords = []
    for i in range(num_objects):
        offset = spread * (i - (num_objects - 1) / 2)
        coords.append([cx + offset, cy])
    return coords

def generate_wedge(num_objects=5, center=(0.5, 0.5), spread=0.1):
    # Wedge: V formasyonuna benzer, fakat açı biraz daha keskin olabilir.
    cx, cy = center
    coords = []
    if num_objects % 2 == 0:
        half = num_objects // 2
        for i in range(half):
            offset = spread * (i + 1)
            coords.append([cx - offset, cy - offset * 0.8])
            coords.append([cx + offset, cy - offset * 0.8])
    else:
        coords.append([cx, cy])
        half = (num_objects - 1) // 2
        for i in range(half):
            offset = spread * (i + 1)
            coords.append([cx - offset, cy - offset * 0.8])
            coords.append([cx + offset, cy - offset * 0.8])
    return coords[:num_objects]

def generate_vee(num_objects=5, center=(0.5, 0.5), spread=0.1):
    # Vee: klasik V formasyonu
    cx, cy = center
    coords = []
    if num_objects % 2 == 0:
        half = num_objects // 2
        for i in range(half):
            offset = spread * (i + 1)
            coords.append([cx - offset, cy - offset])
            coords.append([cx + offset, cy - offset])
    else:
        coords.append([cx, cy])
        half = (num_objects - 1) // 2
        for i in range(half):
            offset = spread * (i + 1)
            coords.append([cx - offset, cy - offset])
            coords.append([cx + offset, cy - offset])
    return coords[:num_objects]

def generate_echelon_right(num_objects=5, center=(0.5, 0.5), spread=0.1):
    # Echelon Right: Sağa doğru kayarak dizilen tanklar
    cx, cy = center
    coords = []
    for i in range(num_objects):
        coords.append([cx + spread * i, cy - spread * i])
    return coords

def generate_herringbone(num_objects=6, center=(0.5, 0.5), spread=0.1):
    # Herringbone: Zigzag şeklinde dizilim
    cx, cy = center
    coords = []
    for i in range(num_objects):
        if i % 2 == 0:
            coords.append([cx - spread * (i / 2), cy - spread * i])
        else:
            coords.append([cx + spread * (i / 2), cy - spread * i])
    return coords

def generate_coil(num_objects=8, center=(0.5, 0.5), radius=0.1):
    # Coil: Spiral (bükümlü) formasyon
    cx, cy = center
    coords = []
    angle_step = 2 * math.pi / num_objects
    for i in range(num_objects):
        angle = i * angle_step
        r = radius * (i + 1) / num_objects
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        coords.append([x, y])
    return coords

def generate_platoon(num_objects=6, center=(0.5, 0.5), spread=0.1):
    # Platoon: Tankların dikdörtgen blok şeklinde dizilmesi
    cx, cy = center
    coords = []
    row1 = num_objects // 2 + num_objects % 2
    row2 = num_objects // 2
    for i in range(row1):
        offset = spread * (i - (row1 - 1) / 2)
        coords.append([cx + offset, cy + spread / 2])
    for i in range(row2):
        offset = spread * (i - (row2 - 1) / 2)
        coords.append([cx + offset, cy - spread / 2])
    return coords

def generate_staggered_column(num_objects=5, center=(0.5, 0.5), spread=0.1):
    # Staggered Column: Tek sütun, ama her satır biraz kaydırılmış
    cx, cy = center
    coords = []
    for i in range(num_objects):
        offset = spread / 2 if i % 2 == 1 else 0
        coords.append([cx + offset, cy - spread * i])
    return coords

def generate_echelon(num_objects=5, center=(0.5, 0.5), spread=0.1):
    # Echelon: Sola doğru kayarak dizilen tanklar
    cx, cy = center
    coords = []
    for i in range(num_objects):
        coords.append([cx - spread * i, cy - spread * i])
    return coords

def generate_column(num_objects=5, center=(0.5, 0.5), spread=0.1):
    # Column: Dikey sütun
    cx, cy = center
    coords = []
    for i in range(num_objects):
        coords.append([cx, cy - spread * i])
    return coords

formation_generators = {
    "Line": generate_line,
    "Wedge": generate_wedge,
    "Vee": generate_vee,
    "Echelon Right": generate_echelon_right,
    "Herringbone": generate_herringbone,
    "Coil": generate_coil,
    "Platoon": generate_platoon,
    "Staggered Column": generate_staggered_column,
    "Echelon": generate_echelon,
    "Column": generate_column
}

# formation_generators = {
#     "0": generate_line,
#     "1": generate_wedge,
#     "2": generate_vee,
#     "3": generate_echelon_right,
#     "4": generate_herringbone,
#     "5": generate_coil,
#     "6": generate_platoon,
#     "7": generate_staggered_column,
#     "8": generate_echelon,
#     "9": generate_column
# }


synthetic_data = []
samples_per_formation = 500  # Her formasyon için 10 örnek oluşturulacak

for formation, generator in formation_generators.items():
    for _ in range(samples_per_formation):
        num_objects = random.choice([5, 6, 7, 8, 9])
        center = (round(random.uniform(0.4, 0.6), 2), round(random.uniform(0.4, 0.6), 2))
        # Coil için farklı parametre: radius
        if formation == "Coil":
            radius = round(random.uniform(0.05, 0.15), 2)
            coords = generator(num_objects=num_objects, center=center, radius=radius)
        else:
            spread = round(random.uniform(0.05, 0.15), 2)
            coords = generator(num_objects=num_objects, center=center, spread=spread)
        sample = {
            "coordinates": coords,
            "classes": ["tank"] * len(coords),
            "formation": formation
        }
        synthetic_data.append(sample)

# Veriyi JSON formatında kaydetme (synthetic_formations.json dosyasına)
with open("synthetic_formations.json", "w") as f:
    json.dump(synthetic_data, f, indent=2)

print("Synthetic dataset for formations created with", len(synthetic_data), "samples!")
