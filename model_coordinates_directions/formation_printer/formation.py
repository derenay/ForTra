import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Veri setini oku
df = pd.read_json("/home/earsal@ETE.local/Desktop/codes/military transformation/tank_formations_mixed.json")


# Her formasyon için çizim yap
for formation, coordinates, angles in zip(df['formation'], df['coordinates'], df['directions']):
    
    x_values = [coord[0] for coord in coordinates]
    y_values = [coord[1] for coord in coordinates]

    plt.figure(figsize=(8, 6))
    plt.scatter(x_values, y_values, color='blue', label='Tanklar')
    plt.plot(x_values, y_values, color='red', linestyle='--', label=f'{formation}')

    # Ok çizimi için ayarlar: arrow_len, ok uzunluğu
    arrow_len = 0.05  # Bu değeri ihtiyaca göre ayarlayabilirsiniz

    for (x, y, angle_norm) in zip(x_values, y_values, angles):
        # Normalize açıları radyana çevir: 0-1 arası -> 0-2π
        angle_rad = angle_norm * 2 * np.pi
        dx = arrow_len * np.cos(angle_rad)
        dy = arrow_len * np.sin(angle_rad)
        plt.arrow(x, y, dx, dy, head_width=0.01, head_length=0.015, fc='green', ec='green')

    plt.title(f"Veri Seti: {formation}")
    plt.xlabel("X Koordinatları")
    plt.ylabel("Y Koordinatları")
    plt.legend()
    plt.grid(True)
    plt.show()
