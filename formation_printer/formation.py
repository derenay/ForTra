import matplotlib.pyplot as plt
import pandas as pd
# Veri setlerini tanımla
df = pd.read_json("dataset/val.json")

# Her bir veri setini çiz
for formation, coordinates in zip(df['formation'], df['coordinates']):


    
    x_values = [coord[0] for coord in coordinates]
    y_values = [coord[1] for coord in coordinates]

    # Çizimi oluştur
    plt.figure(figsize=(8, 6))
    plt.scatter(x_values, y_values, color='blue', label='Tanklar')
    plt.plot(x_values, y_values, color='red', linestyle='--', label=f'{formation}')

    # Başlık ve etiketler
    plt.title(f"Veri Seti : {formation}")
    plt.xlabel("X Koordinatları")
    plt.ylabel("Y Koordinatları")
    plt.legend()
    plt.grid(True)

    # Çizimi göster
    plt.show()