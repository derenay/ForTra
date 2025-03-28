import random
import math
import json

# Düzenli (no noise) formasyon generatorleri
def generate_regular_line(num_objects, center, spread):
    cx, cy = center
    coords = []
    for i in range(num_objects):
        offset = spread * (i - (num_objects-1)/2)
        x = cx + offset
        y = cy  # Sabit y değeri (yatay çizgi)
        coords.append([x, y])
    return coords

def generate_regular_wedge(num_objects, center, spread):
    cx, cy = center
    coords = []
    angle = math.pi/4  # Sabit açı
    per_side = num_objects // 2
    for i in range(per_side):
        left_x = cx - spread*(i+1)*math.cos(angle) 
        left_y = cy - spread*(i+1)*math.sin(angle)
        right_x = cx + spread*(i+1)*math.cos(angle)
        right_y = cy - spread*(i+1)*math.sin(angle)
        coords.append([left_x, left_y])
        coords.append([right_x, right_y])
    return coords[:num_objects]

def generate_regular_vee(num_objects, center, spread):
    cx, cy = center
    coords = []
    angle = math.pi/6  # Sabit açı
    for i in range(num_objects):
        if i % 2 == 0:
            x = cx - spread*(i//2 +1)*math.cos(angle)
            y = cy - spread*(i//2 +1)*math.sin(angle)
        else:
            x = cx + spread*(i//2 +1)*math.cos(angle)
            y = cy - spread*(i//2 +1)*math.sin(angle)
        coords.append([x, y])
    return coords

def generate_regular_echelon_right(num_objects, center, spread):
    cx, cy = center
    coords = []
    step = spread * 0.8
    for i in range(num_objects):
        x = cx + step*i
        y = cy - step*i
        coords.append([x, y])
    return coords

def generate_regular_herringbone(num_objects, center, spread):
    cx, cy = center
    coords = []
    for i in range(num_objects):
        offset = spread*0.5 if i%2 else -spread*0.5
        x = cx + offset
        y = cy - spread*i
        coords.append([x, y])
    return coords

def generate_regular_coil(num_objects, center, radius):
    cx, cy = center
    coords = []
    total_angle = math.pi * 1.5
    for i in range(num_objects):
        angle = total_angle * (i/num_objects)
        r = radius * (i+1)/num_objects * 1.5
        x = cx + r*math.cos(angle) 
        y = cy + r*math.sin(angle)
        coords.append([x, y])
    return coords

def generate_regular_staggered_column(num_objects, center, spread):
    cx, cy = center
    coords = []
    for i in range(num_objects):
        offset = spread*0.3 if i%2 else -spread*0.3
        x = cx + offset
        y = cy - spread*i
        coords.append([x, y])
    return coords

def generate_regular_echelon_left(num_objects, center, spread):
    cx, cy = center
    coords = []
    step = spread * 0.8
    for i in range(num_objects):
        x = cx - step*i
        y = cy - step*i
        coords.append([x, y])
    return coords

def generate_regular_column(num_objects, center, spread):
    cx, cy = center
    coords = []
    for i in range(num_objects):
        x = cx
        y = cy - spread*i
        coords.append([x, y])
    return coords

# Rastgele (noisy) formasyon generatorleri (mevcut kodunuzda olanlar)
def generate_line(num_objects, center, spread):
    cx, cy = center
    coords = []
    angle = random.uniform(-math.pi/18, math.pi/18)
    for i in range(num_objects):
        offset = spread * (i - (num_objects-1)/2)
        x = cx + offset * math.cos(angle)
        y = cy + offset * math.sin(angle)
        coords.append([
            x + random.uniform(-spread*0.05, spread*0.05),
            y + random.uniform(-spread*0.05, spread*0.05)
        ])
    return coords

def generate_wedge(num_objects, center, spread):
    cx, cy = center
    coords = []
    angle = math.pi/4
    per_side = num_objects // 2
    for i in range(per_side):
        left_x = cx - spread*(i+1)*math.cos(angle) 
        left_y = cy - spread*(i+1)*math.sin(angle)
        right_x = cx + spread*(i+1)*math.cos(angle)
        right_y = cy - spread*(i+1)*math.sin(angle)
        coords.append([left_x + random.uniform(-spread*0.1, spread*0.1),
                       left_y + random.uniform(-spread*0.1, spread*0.1)])
        coords.append([right_x + random.uniform(-spread*0.1, spread*0.1),
                       right_y + random.uniform(-spread*0.1, spread*0.1)])
    return coords[:num_objects]

def generate_vee(num_objects, center, spread):
    cx, cy = center
    coords = []
    angle = math.pi/6
    for i in range(num_objects):
        if i % 2 == 0:
            x = cx - spread*(i//2 +1)*math.cos(angle)
            y = cy - spread*(i//2 +1)*math.sin(angle)
        else:
            x = cx + spread*(i//2 +1)*math.cos(angle)
            y = cy - spread*(i//2 +1)*math.sin(angle)
        coords.append([x + random.uniform(-spread*0.1, spread*0.1),
                       y + random.uniform(-spread*0.1, spread*0.1)])
    return coords

def generate_echelon_right(num_objects, center, spread):
    cx, cy = center
    coords = []
    step = spread * 0.8
    for i in range(num_objects):
        x = cx + step*i + random.uniform(-spread*0.1, spread*0.1)
        y = cy - step*i + random.uniform(-spread*0.1, spread*0.1)
        coords.append([x, y])
    return coords

def generate_herringbone(num_objects, center, spread):
    cx, cy = center
    coords = []
    for i in range(num_objects):
        offset = spread*0.5 if i%2 else -spread*0.5
        x = cx + offset + random.uniform(-spread*0.1, spread*0.1)
        y = cy - spread*i + random.uniform(-spread*0.1, spread*0.1)
        coords.append([x, y])
    return coords

def generate_coil(num_objects, center, radius):
    cx, cy = center
    coords = []
    total_angle = math.pi * 1.5
    for i in range(num_objects):
        angle = total_angle * (i/num_objects)
        r = radius * (i+1)/num_objects * 1.5
        x = cx + r*math.cos(angle) 
        y = cy + r*math.sin(angle)
        coords.append([x + random.uniform(-radius*0.05, radius*0.05),
                       y + random.uniform(-radius*0.05, radius*0.05)])
    return coords

def generate_staggered_column(num_objects, center, spread):
    cx, cy = center
    coords = []
    for i in range(num_objects):
        offset = spread*0.3 if i%2 else -spread*0.3
        x = cx + offset + random.uniform(-spread*0.1, spread*0.1)
        y = cy - spread*i + random.uniform(-spread*0.1, spread*0.1)
        coords.append([x, y])
    return coords

def generate_echelon_left(num_objects, center, spread):
    cx, cy = center
    coords = []
    step = spread * 0.8
    for i in range(num_objects):
        x = cx - step*i + random.uniform(-spread*0.1, spread*0.1)
        y = cy - step*i + random.uniform(-spread*0.1, spread*0.1)
        coords.append([x, y])
    return coords

def generate_column(num_objects, center, spread):
    cx, cy = center
    coords = []
    for i in range(num_objects):
        x = cx + random.uniform(-spread*0.1, spread*0.1)
        y = cy - spread*i + random.uniform(-spread*0.1, spread*0.1)
        coords.append([x, y])
    return coords

# Parametreler ve generator mapping'leri
formation_params = {
    "0": {"num_objects": (4, 12), "spread": (0.05, 0.10)},
    "1": {"num_objects": (4, 12), "spread": (0.05, 0.10)},
    "2": {"num_objects": (3, 12), "spread": (0.05, 0.10)},
    "3": {"num_objects": (3, 10), "spread": (0.05, 0.10)},
    "4": {"num_objects": (4, 12), "spread": (0.05, 0.10)},
    "5": {"num_objects": (4, 15), "radius": (0.05, 0.10)},
    "6": {"num_objects": (4, 10), "spread": (0.05, 0.1)},
    "7": {"num_objects": (4, 10), "spread": (0.05, 0.10)},
    "8": {"num_objects": (4, 10), "spread": (0.05, 0.10)},
}

formation_generators = {
    "0": generate_line,# line yan yana gittikleri
    "1": generate_wedge, # wedge te gene v şeklinde ama ateş gücü öne bakıyor ters v şeklinde olmalı
    "2": generate_vee, # vee de v şeklinde olması durumu
    "3": generate_echelon_right,
    "4": generate_herringbone, # burda ise tankların konumu staggered_column gibi ama tanklar durmuş ve 360 etrafa bakıyor
    "5": generate_coil, # coil de tanklar kare çeklinde duurp sağdakinin sağa öndekinin öne arkadakinin arkaya bakma şekli olucak
    "6": generate_staggered_column, # gene arka arkaya ama çapraz giden
    "7": generate_echelon_left,
    "8": generate_column, # colon arka arkaya gittikleri 
    "9": generate_echelon # wedgeye benziyor ama tankların önden çok daha fazla yanlara ateş gücü vermesi
}

regular_generators = {
    "0": generate_regular_line,
    "1": generate_regular_wedge,
    "2": generate_regular_vee,
    "3": generate_regular_echelon_right,
    "4": generate_regular_herringbone,
    "5": generate_regular_coil,
    "6": generate_regular_staggered_column,
    "7": generate_regular_echelon_left,
    "8": generate_regular_column,
}

# Veri üretimi
synthetic_data = []
samples_per_formation = 10000
tank_formations = list(formation_generators.keys())

for formation in tank_formations:
    noisy_generator = formation_generators[formation]
    regular_generator = regular_generators[formation]
    params = formation_params[formation]
    
    for _ in range(samples_per_formation):
        num_objects = random.randint(params["num_objects"][0], params["num_objects"][1])
        center = (round(random.uniform(0.35, 0.65), 2),
                  round(random.uniform(0.35, 0.65), 2))
        
        # %70 rastgele, %30 düzenli veri üret
        if random.random() < 0.7:
            # Rastgele veri
            if formation == "5":
                radius = round(random.uniform(params["radius"][0], params["radius"][1]), 2)
                coords = noisy_generator(num_objects, center, radius)
            else:
                spread = round(random.uniform(params["spread"][0], params["spread"][1]), 2)
                coords = noisy_generator(num_objects, center, spread)
        else:
            # Düzenli veri
            if formation == "5":
                radius = round(random.uniform(params["radius"][0], params["radius"][1]), 2)
                coords = regular_generator(num_objects, center, radius)
            else:
                spread = round(random.uniform(params["spread"][0], params["spread"][1]), 2)
                coords = regular_generator(num_objects, center, spread)
        
        sample = {
            "coordinates": [[round(x, 3) for x in point] for point in coords],
            "classes": [0] * len(coords),
            "formation": int(formation)
        }
        synthetic_data.append(sample)



# JSON olarak kaydet
with open("tank_formations_mixed.json", "w") as f:
    json.dump(synthetic_data, f, indent=2)

print(f"Veri seti oluşturuldu: {len(synthetic_data)} örnek")