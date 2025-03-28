import random
import json
import math

# Formation-Specific Generators with Dynamic Variations
def generate_line(num_objects, center, spread):
    cx, cy = center
    coords = []
    base_angle = random.uniform(-math.pi/12, math.pi/12)  # Slight curvature
    for i in range(num_objects):
        offset = spread * (i - (num_objects-1)/2)
        x = cx + offset * math.cos(base_angle)
        y = cy + offset * math.sin(base_angle)
        coords.append([
            x + random.uniform(-spread*0.1, spread*0.1),
            y + random.uniform(-spread*0.1, spread*0.1)
        ])
    return coords

def generate_wedge(num_objects, center, spread):
    cx, cy = center
    coords = []
    angle = random.uniform(math.pi/6, math.pi/3)  # 30-60 degree V
    for i in range(num_objects):
        if i % 2 == 0:
            x = cx - spread*(i+1)*math.cos(angle)
            y = cy - spread*(i+1)*math.sin(angle)
        else:
            x = cx + spread*(i+1)*math.cos(angle)
            y = cy - spread*(i+1)*math.sin(angle)
        coords.append([
            x + random.uniform(-spread*0.2, spread*0.2),
            y + random.uniform(-spread*0.2, spread*0.2)
        ])
    return coords

def generate_vee(num_objects, center, spread):
    cx, cy = center
    coords = []
    angle = math.pi/4  # 45 degree V
    for i in range(num_objects):
        if i % 2 == 0:
            coords.append([
                cx - spread*(i+1)*math.cos(angle) + random.uniform(-spread*0.15, spread*0.15),
                cy - spread*(i+1)*math.sin(angle) + random.uniform(-spread*0.15, spread*0.15)
            ])
        else:
            coords.append([
                cx + spread*(i+1)*math.cos(angle) + random.uniform(-spread*0.15, spread*0.15),
                cy - spread*(i+1)*math.sin(angle) + random.uniform(-spread*0.15, spread*0.15)
            ])
    return coords

def generate_echelon_right(num_objects, center, spread):
    cx, cy = center
    coords = []
    for i in range(num_objects):
        x = cx + spread*i + random.uniform(-spread*0.3, spread*0.3)
        y = cy - spread*i + random.uniform(-spread*0.3, spread*0.3)
        coords.append([x, y])
    return coords

def generate_herringbone(num_objects, center, spread):
    cx, cy = center
    coords = []
    for i in range(num_objects):
        offset = random.choice([-1, 1]) * spread * 0.5
        coords.append([
            cx + (spread*i*0.5 if i%2==0 else -spread*i*0.5) + offset,
            cy - spread*i + random.uniform(-spread*0.2, spread*0.2)
        ])
    return coords

def generate_coil(num_objects, center, radius):
    cx, cy = center
    coords = []
    total_angle = random.uniform(math.pi*1.5, math.pi*3)
    for i in range(num_objects):
        angle = total_angle * (i/num_objects)
        r = radius * (i+1)/num_objects
        coords.append([
            cx + r*math.cos(angle) + random.uniform(-radius*0.1, radius*0.1),
            cy + r*math.sin(angle) + random.uniform(-radius*0.1, radius*0.1)
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
            coords.append([
                cx + (j - (cols-1)/2)*spread + random.uniform(-spread*0.2, spread*0.2),
                cy + (i - (rows-1)/2)*spread + random.uniform(-spread*0.2, spread*0.2)
            ])
    return coords

def generate_staggered_column(num_objects, center, spread):
    cx, cy = center
    coords = []
    for i in range(num_objects):
        offset = random.choice([-1, 1]) * spread*0.3 if i%2 else 0
        coords.append([
            cx + offset + random.uniform(-spread*0.1, spread*0.1),
            cy - spread*i + random.uniform(-spread*0.2, spread*0.2)
        ])
    return coords

def generate_echelon_left(num_objects, center, spread):
    cx, cy = center
    coords = []
    for i in range(num_objects):
        x = cx - spread*i + random.uniform(-spread*0.3, spread*0.3)
        y = cy - spread*i + random.uniform(-spread*0.3, spread*0.3)
        coords.append([x, y])
    return coords

def generate_column(num_objects, center, spread):
    cx, cy = center
    coords = []
    for i in range(num_objects):
        coords.append([
            cx + random.uniform(-spread*0.2, spread*0.2),
            cy - spread*i + random.uniform(-spread*0.2, spread*0.2)
        ])
    return coords

def generate_diamond(num_objects, center, spread):
    cx, cy = center
    coords = []
    layers = math.ceil((math.sqrt(num_objects*2)-1)/2)
    for layer in range(layers+1):
        for _ in range(4):
            if len(coords) >= num_objects:
                break
            angle = random.uniform(0, math.pi*2)
            x = cx + spread*layer*math.cos(angle)
            y = cy + spread*layer*math.sin(angle)
            coords.append([
                x + random.uniform(-spread*0.2, spread*0.2),
                y + random.uniform(-spread*0.2, spread*0.2)
            ])
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
                        cx + spread*(j - (side-1)/2) + random.uniform(-spread*0.2, spread*0.2),
                        cy + spread*(i - (side-1)/2) + random.uniform(-spread*0.2, spread*0.2)
                    ])
    return coords

def generate_arrowhead(num_objects, center, spread):
    cx, cy = center
    coords = []
    tip = [cx + random.uniform(-spread*0.5, spread*0.5), cy - spread*1.5]
    coords.append(tip)
    for i in range(1, num_objects):
        angle = math.pi/3 * (1 if i%2==0 else -1)
        x = cx + spread*i*0.8*math.cos(angle) + random.uniform(-spread*0.3, spread*0.3)
        y = cy - spread*i*0.8*math.sin(angle) + random.uniform(-spread*0.3, spread*0.3)
        coords.append([x, y])
    return coords

def generate_cross(num_objects, center, spread):
    cx, cy = center
    coords = []
    vertical = num_objects // 2
    horizontal = num_objects - vertical
    for i in range(-vertical//2, vertical//2 +1):
        coords.append([cx + random.uniform(-spread*0.3, spread*0.3),
                      cy + spread*i + random.uniform(-spread*0.3, spread*0.3)])
    for i in range(1, horizontal//2 +1):
        coords.append([cx + spread*i + random.uniform(-spread*0.3, spread*0.3),
                      cy + random.uniform(-spread*0.3, spread*0.3)])
        coords.append([cx - spread*i + random.uniform(-spread*0.3, spread*0.3),
                      cy + random.uniform(-spread*0.3, spread*0.3)])
    return coords

def generate_t_formation(num_objects, center, spread):
    cx, cy = center
    coords = []
    # Vertical stem
    stem = num_objects // 3
    for i in range(stem):
        coords.append([cx + random.uniform(-spread*0.3, spread*0.3),
                      cy - spread*i + random.uniform(-spread*0.3, spread*0.3)])
    # Horizontal bar
    bar = num_objects - stem
    for i in range(-(bar//2), bar//2 +1):
        if len(coords) < num_objects:
            coords.append([cx + spread*i + random.uniform(-spread*0.3, spread*0.3),
                          cy - spread*(stem-1) + random.uniform(-spread*0.3, spread*0.3)])
    return coords

def generate_y_formation(num_objects, center, spread):
    cx, cy = center
    coords = []
    angles = [math.pi/6, -math.pi/6, math.pi/2]
    for i in range(num_objects):
        angle = angles[i%3] + random.uniform(-math.pi/12, math.pi/12)
        length = spread * (i//3 + 1)
        coords.append([
            cx + length*math.cos(angle) + random.uniform(-spread*0.2, spread*0.2),
            cy + length*math.sin(angle) + random.uniform(-spread*0.2, spread*0.2)
        ])
    return coords

def generate_zigzag(num_objects, center, spread):
    cx, cy = center
    coords = []
    for i in range(num_objects):
        offset = spread*0.5 if i%2 else -spread*0.5
        coords.append([
            cx + offset + random.uniform(-spread*0.2, spread*0.2),
            cy - spread*i + random.uniform(-spread*0.2, spread*0.2)
        ])
    return coords

def generate_checkerboard(num_objects, center, spread):
    cx, cy = center
    coords = []
    grid_size = math.ceil(math.sqrt(num_objects))
    for i in range(grid_size):
        for j in range(grid_size):
            if (i + j) % 2 == 0 and len(coords) < num_objects:
                coords.append([
                    cx + spread*(j - (grid_size-1)/2) + random.uniform(-spread*0.3, spread*0.3),
                    cy + spread*(i - (grid_size-1)/2) + random.uniform(-spread*0.3, spread*0.3)
                ])
    return coords

def generate_fan(num_objects, center, spread):
    cx, cy = center
    coords = []
    start_angle = random.uniform(-math.pi/4, math.pi/4)
    angle_range = random.uniform(math.pi*0.5, math.pi)
    for i in range(num_objects):
        angle = start_angle + angle_range*(i/(num_objects-1))
        coords.append([
            cx + spread*math.cos(angle) + random.uniform(-spread*0.2, spread*0.2),
            cy + spread*math.sin(angle) + random.uniform(-spread*0.2, spread*0.2)
        ])
    return coords

def generate_hollow_square(num_objects, center, spread):
    cx, cy = center
    coords = []
    side_length = math.ceil(math.sqrt(num_objects)) // 2
    sides = 4
    per_side = (num_objects - 4) // sides  # Keep corners separate
    # Add corners first
    corners = [
        [cx - spread, cy - spread],
        [cx + spread, cy - spread],
        [cx + spread, cy + spread],
        [cx - spread, cy + spread]
    ]
    coords.extend(corners[:min(4, num_objects)])
    remaining = max(0, num_objects - 4)
    for _ in range(remaining):
        side = random.randint(0, 3)
        if side == 0:  # Top
            x = cx - spread + random.uniform(spread*0.2, spread*1.8)
            y = cy - spread + random.uniform(-spread*0.2, spread*0.2)
        elif side == 1:  # Right
            x = cx + spread + random.uniform(-spread*0.2, spread*0.2)
            y = cy - spread + random.uniform(spread*0.2, spread*1.8)
        elif side == 2:  # Bottom
            x = cx - spread + random.uniform(spread*0.2, spread*1.8)
            y = cy + spread + random.uniform(-spread*0.2, spread*0.2)
        else:  # Left
            x = cx - spread + random.uniform(-spread*0.2, spread*0.2)
            y = cy - spread + random.uniform(spread*0.2, spread*1.8)
        coords.append([x, y])
    return coords


# Formation parameters with dynamic ranges
formation_params = {
    "Line": {
        "num_objects": (5, 9),
        "spread": (0.08, 0.15)
    },
    "Wedge": {
        "num_objects": (6, 10),
        "spread": (0.1, 0.2)
    },
    "Vee": {
        "num_objects": (5, 9),
        "spread": (0.1, 0.2)
    },
    "Echelon Right": {
        "num_objects": (5, 8),
        "spread": (0.08, 0.15)
    },
    "Herringbone": {
        "num_objects": (6, 10),
        "spread": (0.1, 0.2)
    },
    "Coil": {
        "num_objects": (8, 12),
        "radius": (0.1, 0.2)
    },
    "Platoon": {
        "num_objects": (6, 12),
        "spread": (0.1, 0.2)
    },
    "Staggered Column": {
        "num_objects": (5, 9),
        "spread": (0.08, 0.15)
    },
    "Echelon Left": {
        "num_objects": (5, 8),
        "spread": (0.08, 0.15)
    },
    "Column": {
        "num_objects": (5, 9),
        "spread": (0.05, 0.1)
    },
    "Diamond": {
        "num_objects": (7, 12),
        "spread": (0.1, 0.2)
    },
    "Box": {
        "num_objects": (8, 16),
        "spread": (0.15, 0.25)
    },
    "Arrowhead": {
        "num_objects": (7, 12),
        "spread": (0.15, 0.25)
    },
    "Cross": {
        "num_objects": (5, 10),
        "spread": (0.1, 0.2)
    },
    "T-Formation": {
        "num_objects": (6, 10),
        "spread": (0.1, 0.2)
    },
    "Y Formation": {
        "num_objects": (6, 9),
        "spread": (0.1, 0.2)
    },
    "Zigzag": {
        "num_objects": (6, 10),
        "spread": (0.1, 0.2)
    },
    "Checkerboard": {
        "num_objects": (9, 16),
        "spread": (0.15, 0.25)
    },
    "Fan": {
        "num_objects": (8, 12),
        "spread": (0.15, 0.25)
    },
    "Hollow Square": {
        "num_objects": (8, 16),
        "spread": (0.2, 0.3)
    },
}

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
samples_per_formation = 10


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