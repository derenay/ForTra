import random
import math
import json

# Tank formasyonları için generatör fonksiyonları
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

# Sadece tank formasyonlarını içeren parametreler
formation_params = {
    "Line": {"num_objects": (8, 12), "spread": (0.05, 0.20)},
    "Wedge": {"num_objects": (8, 12), "spread": (0.05, 0.20)},
    "Vee": {"num_objects": (8, 12), "spread": (0.05, 0.20)},
    "Echelon Right": {"num_objects": (6, 10), "spread": (0.05, 0.20)},
    "Herringbone": {"num_objects": (8, 12), "spread": (0.05, 0.20)},
    "Coil": {"num_objects": (10, 15), "radius": (0.05, 0.20)},
    "Staggered Column": {"num_objects": (6, 10), "spread": (0.05, 0.20)},
    "Echelon Left": {"num_objects": (6, 10), "spread": (0.05, 0.20)},
    "Column": {"num_objects": (6, 10), "spread": (0.05, 0.20)},
}

# Formasyon jeneratör mapping'i
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
}

# Veri üretimi
synthetic_data = []
samples_per_formation = 200
tank_formations = list(formation_generators.keys())

for formation in tank_formations:
    generator = formation_generators[formation]
    params = formation_params[formation]
    
    for _ in range(samples_per_formation):
        num_objects = random.randint(params["num_objects"][0], params["num_objects"][1])
        center = (round(random.uniform(0.35, 0.65), 2),
                  round(random.uniform(0.35, 0.65), 2))
        
        if formation == "Coil":
            radius = round(random.uniform(params["radius"][0], params["radius"][1]), 2)
            coords = generator(num_objects, center, radius)
        else:
            spread = round(random.uniform(params["spread"][0], params["spread"][1]), 2)
            coords = generator(num_objects, center, spread)
        
        sample = {
            "coordinates": [[round(x, 3) for x in point] for point in coords],
            "classes": ["tank"] * len(coords),
            "formation": formation
        }
        synthetic_data.append(sample)

# JSON olarak kaydetme
with open("tank_formations.json", "w") as f:
    json.dump(synthetic_data, f, indent=2)

print(f"Veri seti oluşturuldu: {len(synthetic_data)} örnek")
