import random
import math
import json

# Merkezi konum ve yayılım kontrollü generatörler
def generate_line(num_objects, center, spread):
    cx, cy = center
    coords = []
    angle = random.uniform(-math.pi/18, math.pi/18)  # Daha düz çizgi
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
    angle = math.pi/4  # 45 derece V
    per_side = num_objects // 2
    for i in range(per_side):
        left_x = cx - spread*(i+1)*math.cos(angle) 
        left_y = cy - spread*(i+1)*math.sin(angle)
        right_x = cx + spread*(i+1)*math.cos(angle)
        right_y = cy - spread*(i+1)*math.sin(angle)
        coords.append([
            left_x + random.uniform(-spread*0.1, spread*0.1),
            left_y + random.uniform(-spread*0.1, spread*0.1)
        ])
        coords.append([
            right_x + random.uniform(-spread*0.1, spread*0.1),
            right_y + random.uniform(-spread*0.1, spread*0.1)
        ])
    return coords[:num_objects]

def generate_vee(num_objects, center, spread):
    cx, cy = center
    coords = []
    angle = math.pi/6  # 30 derece V
    for i in range(num_objects):
        if i % 2 == 0:
            x = cx - spread*(i//2 +1)*math.cos(angle)
            y = cy - spread*(i//2 +1)*math.sin(angle)
        else:
            x = cx + spread*(i//2 +1)*math.cos(angle)
            y = cy - spread*(i//2 +1)*math.sin(angle)
        coords.append([
            x + random.uniform(-spread*0.1, spread*0.1),
            y + random.uniform(-spread*0.1, spread*0.1)
        ])
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
    total_angle = math.pi * 1.5  # 270 derece spiral
    for i in range(num_objects):
        angle = total_angle * (i/num_objects)
        r = radius * (i+1)/num_objects * 1.5
        x = cx + r*math.cos(angle) 
        y = cy + r*math.sin(angle)
        coords.append([
            x + random.uniform(-radius*0.05, radius*0.05),
            y + random.uniform(-radius*0.05, radius*0.05)
        ])
    return coords

def generate_platoon(num_objects, center, spread):
    cx, cy = center
    coords = []
    rows = math.ceil(math.sqrt(num_objects))
    cols = math.ceil(num_objects/rows)
    for i in range(rows):
        for j in range(cols):
            if i*cols + j >= num_objects:
                break
            x = cx + spread*(j - cols/2) 
            y = cy + spread*(i - rows/2)
            coords.append([
                x + random.uniform(-spread*0.1, spread*0.1),
                y + random.uniform(-spread*0.1, spread*0.1)
            ])
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

def generate_diamond(num_objects, center, spread):
    cx, cy = center
    coords = []
    layers = math.ceil(math.sqrt(num_objects))
    for layer in range(layers):
        for _ in range(4):
            angle = math.pi/2 * random.randint(0,3)
            x = cx + spread*layer*math.cos(angle)
            y = cy + spread*layer*math.sin(angle)
            coords.append([
                x + random.uniform(-spread*0.15, spread*0.15),
                y + random.uniform(-spread*0.15, spread*0.15)
            ])
            if len(coords) >= num_objects:
                return coords
    return coords

def generate_box(num_objects, center, spread):
    cx, cy = center
    coords = []
    side = math.ceil(math.sqrt(num_objects))
    for i in range(side):
        for j in range(side):
            if i == 0 or i == side-1 or j == 0 or j == side-1:
                if len(coords) < num_objects:
                    coords.append([
                        cx + spread*(j - side/2) + random.uniform(-spread*0.15, spread*0.15),
                        cy + spread*(i - side/2) + random.uniform(-spread*0.15, spread*0.15)
                    ])
    return coords

def generate_arrowhead(num_objects, center, spread):
    cx, cy = center
    coords = []
    # Ok ucu
    coords.append([cx, cy - spread*2])
    # Sol kanat
    for i in range(1, (num_objects-1)//2 +1):
        x = cx - spread*i
        y = cy - spread*(2 - i)
        coords.append([x + random.uniform(-spread*0.1, spread*0.1),
                      y + random.uniform(-spread*0.1, spread*0.1)])
    # Sağ kanat
    for i in range(1, (num_objects-1)//2 +1):
        x = cx + spread*i
        y = cy - spread*(2 - i)
        coords.append([x + random.uniform(-spread*0.1, spread*0.1),
                      y + random.uniform(-spread*0.1, spread*0.1)])
    return coords[:num_objects]

def generate_cross(num_objects, center, spread):
    cx, cy = center
    coords = []
    vertical = num_objects // 2
    horizontal = num_objects - vertical
    # Dikey çizgi
    for i in range(-vertical//2, vertical//2 +1):
        coords.append([cx + random.uniform(-spread*0.1, spread*0.1),
                      cy + spread*i + random.uniform(-spread*0.1, spread*0.1)])
    # Yatay çizgi
    for i in range(1, horizontal//2 +1):
        coords.append([cx + spread*i + random.uniform(-spread*0.1, spread*0.1),
                      cy + random.uniform(-spread*0.1, spread*0.1)])
        coords.append([cx - spread*i + random.uniform(-spread*0.1, spread*0.1),
                      cy + random.uniform(-spread*0.1, spread*0.1)])
    return coords

def generate_t_formation(num_objects, center, spread):
    cx, cy = center
    coords = []
    # Dikey gövde
    stem = num_objects // 3
    for i in range(stem):
        coords.append([cx + random.uniform(-spread*0.1, spread*0.1),
                      cy - spread*i + random.uniform(-spread*0.1, spread*0.1)])
    # Yatay çubuk
    bar = num_objects - stem
    for i in range(-(bar//2), bar//2 +1):
        if len(coords) < num_objects:
            coords.append([cx + spread*i + random.uniform(-spread*0.1, spread*0.1),
                          cy - spread*(stem-1) + random.uniform(-spread*0.1, spread*0.1)])
    return coords

def generate_y_formation(num_objects, center, spread):
    cx, cy = center
    coords = []
    angles = [math.pi/6, -math.pi/6, math.pi/2]  # 30°, -30°, 90°
    per_arm = num_objects // 3
    remainder = num_objects % 3
    
    for i in range(per_arm + (1 if remainder else 0)):
        for angle in angles:
            if len(coords) >= num_objects:
                break
            length = spread * (i+1) * 1.2
            x = cx + length * math.cos(angle)
            y = cy + length * math.sin(angle)
            coords.append([
                x + random.uniform(-spread*0.05, spread*0.05),
                y + random.uniform(-spread*0.05, spread*0.05)
            ])
    return coords

def generate_zigzag(num_objects, center, spread):
    cx, cy = center
    coords = []
    for i in range(num_objects):
        offset = spread*0.6 if i%2 else -spread*0.6
        x = cx + offset + random.uniform(-spread*0.1, spread*0.1)
        y = cy - spread*i + random.uniform(-spread*0.1, spread*0.1)
        coords.append([x, y])
    return coords

def generate_checkerboard(num_objects, center, spread):
    cx, cy = center
    coords = []
    grid_size = math.ceil(math.sqrt(num_objects))
    for i in range(grid_size):
        for j in range(grid_size):
            if (i + j) % 2 == 0 and len(coords) < num_objects:
                coords.append([
                    cx + spread*(j - grid_size/2) + random.uniform(-spread*0.15, spread*0.15),
                    cy + spread*(i - grid_size/2) + random.uniform(-spread*0.15, spread*0.15)
                ])
    return coords

def generate_fan(num_objects, center, spread):
    cx, cy = center
    coords = []
    start_angle = math.pi/4
    angle_range = math.pi/2
    for i in range(num_objects):
        angle = start_angle + angle_range*(i/(num_objects-1)) - angle_range/2
        x = cx + spread*1.5*math.cos(angle)
        y = cy + spread*1.5*math.sin(angle)
        coords.append([
            x + random.uniform(-spread*0.1, spread*0.1),
            y + random.uniform(-spread*0.1, spread*0.1)
        ])
    return coords

def generate_hollow_square(num_objects, center, spread):
    cx, cy = center
    coords = []
    side_length = math.ceil(math.sqrt(num_objects)) // 2
    sides = 4
    per_side = (num_objects - 4) // sides
    
    # Köşeler
    corners = [
        [cx - spread, cy - spread],
        [cx + spread, cy - spread],
        [cx + spread, cy + spread],
        [cx - spread, cy + spread]
    ]
    coords.extend(corners[:min(4, num_objects)])
    
    # Kenarlar
    for _ in range(per_side*4):
        if len(coords) >= num_objects:
            break
        side = random.randint(0, 3)
        if side == 0:  # Üst kenar
            x = cx - spread + random.uniform(spread*0.2, spread*1.8)
            y = cy - spread + random.uniform(-spread*0.1, spread*0.1)
        elif side == 1:  # Sağ kenar
            x = cx + spread + random.uniform(-spread*0.1, spread*0.1)
            y = cy - spread + random.uniform(spread*0.2, spread*1.8)
        elif side == 2:  # Alt kenar
            x = cx - spread + random.uniform(spread*0.2, spread*1.8)
            y = cy + spread + random.uniform(-spread*0.1, spread*0.1)
        else:  # Sol kenar
            x = cx - spread + random.uniform(-spread*0.1, spread*0.1)
            y = cy - spread + random.uniform(spread*0.2, spread*1.8)
        coords.append([x, y])
    return coords

# Parametrelerde iyileştirmeler
formation_params = {
    "Line": {"num_objects": (8, 12), "spread": (0.05, 0.20)},
    "Wedge": {"num_objects": (8, 12), "spread": (0.05, 0.20)},
    "Vee": {"num_objects": (8, 12), "spread": (0.05, 0.20)},
    "Echelon Right": {"num_objects": (6, 10), "spread": (0.05, 0.20)},
    "Herringbone": {"num_objects": (8, 12), "spread": (0.05, 0.20)},
    "Coil": {"num_objects": (10, 15), "radius": (0.05, 0.20)},
    "Platoon": {"num_objects": (8, 16), "spread": (0.05, 0.20)},
    "Staggered Column": {"num_objects": (6, 10), "spread": (0.05, 0.20)},
    "Echelon Left": {"num_objects": (6, 10), "spread": (0.05, 0.20)},
    "Column": {"num_objects": (6, 10), "spread": (0.05, 0.20)},
    "Diamond": {"num_objects": (10, 16), "spread": (0.05, 0.20)},
    "Box": {"num_objects": (12, 20), "spread": (0.05, 0.20)},
    "Arrowhead": {"num_objects": (10, 15), "spread": (0.05, 0.20)},
    "Cross": {"num_objects": (8, 12), "spread": (0.05, 0.20)},
    "T-Formation": {"num_objects": (8, 12), "spread": (0.05, 0.20)},
    "Y Formation": {"num_objects": (9, 15), "spread": (0.05, 0.20)},
    "Zigzag": {"num_objects": (8, 12), "spread": (0.05, 0.20)},
    "Checkerboard": {"num_objects": (16, 25), "spread": (0.05, 0.20)},
    "Fan": {"num_objects": (10, 15), "spread": (0.05, 0.20)},
    "Hollow Square": {"num_objects": (12, 20), "spread": (0.05, 0.20)}
}

# Geri kalan kod (data üretimi ve kayıt kısmı) aynı kalıyor

# Mapping from formation name to generator function
formation_generators = {
    "Line": generate_line,
    "Wedge": generate_wedge,
    "Vee": generate_vee,
    "Echelon Right": generate_echelon_right,
    "Herringbone": generate_herringbone,
    "Coil": generate_coil,
    "Platoon": generate_platoon,
    "Staggered Column": generate_staggered_column,
    "Echelon Left": generate_echelon_left,
    "Column": generate_column,
    "Diamond": generate_diamond,
    "Box": generate_box,
    "Arrowhead": generate_arrowhead,
    "Cross": generate_cross,
    "T-Formation": generate_t_formation,
    "Y Formation": generate_y_formation,
    "Zigzag": generate_zigzag,
    "Checkerboard": generate_checkerboard,
    "Fan": generate_fan,
    "Hollow Square": generate_hollow_square,
}

# Generate synthetic data
synthetic_data = []
samples_per_formation = 100


for formation, generator in formation_generators.items():
    params = formation_params[formation]
    num_range = params["num_objects"]
    
    for _ in range(samples_per_formation):
        # Randomize parameters
        num_objects = random.randint(num_range[0], num_range[1])
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

# Save to JSON
with open("synthetic_formations.json", "w") as f:
    json.dump(synthetic_data, f, indent=2)

print(f"Dataset created with {len(synthetic_data)} samples")