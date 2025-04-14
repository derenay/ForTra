# server.py
# Current Date: 2025-04-09

import torch
# import pandas as pd # Pandas'a bu dosyada doğrudan gerek kalmadı
# import numpy as np # Numpy'a bu dosyada doğrudan gerek kalmadı
from typing import List, Dict, Any
import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS # CORS'u kullanmak için import et

# --- Model ve Yardımcı Fonksiyonları Import Et ---
# Bu dosyaların server.py ile aynı dizinde veya Python path'inde olduğundan emin olun
try:
    from model import HierarchicalFormationTransformer
    # model_tools.py içinde model ağırlıklarını yükleyen bir fonksiyon olduğunu varsayıyoruz
    from model_tools import load_model
except ImportError as e:
    print(f"HATA: Gerekli modüller import edilemedi: {e}")
    print("Lütfen 'model.py' ve 'model_tools.py' dosyalarının doğru yerde olduğundan emin olun.")
    exit()

# --- Sabitler ve Map'ler (Eğitimde kullanılanlarla aynı olmalı) ---
# Bu değerler, modelinizin eğitildiği verilere ve konfigürasyona göre ayarlanmalıdır.
CLASS_TO_IDX = {
    'tank': 0,
    # Eğitimde başka sınıflar kullandıysanız buraya ekleyin
    # 'ship': 1,
}

FORMATION_TO_IDX = {
    "Line": 0, "Wedge": 1, "Vee": 2, "Herringbone": 3,
    "Coil": 4, "Staggered Column": 5, "Column": 6, "Echelon": 7
    # Eğitimdeki tüm formasyonlar burada olmalı
}
# Index'ten isme çevirmek için ters map oluştur
IDX_TO_FORMATION = {i: name for name, i in FORMATION_TO_IDX.items()}

# --- MODEL HİPERPARAMETRELERİ (Eğitimde kullanılanlarla aynı olmalı) ---
# Bu değerler, yükleyeceğiniz modelin eğitildiği ayarlarla eşleşmelidir.
MODEL_CONFIG = {
    'class_embed_dim': 32,
    'direction_dim': 1,            # Verinize göre ayarlayın (1: açı, 2: vektör)
    'stage_dims': [256, 128],      # Önceki önerilen dengeli ayar
    'num_heads': 8,
    'num_layers': [6, 6],
    'dropout_stages': [0.2, 0.1],
    'use_adapter': True,
    'adapter_dim': 32,
    'pos_type': 'learnable',
    'max_len': 50,
    'ffn_ratio': 4,
    'coord_dim': 2,                # Genellikle 2D koordinat
    # Bu değerler map'lerden otomatik alınacak:
    # 'class_vocab_size': len(CLASS_TO_IDX),
    # 'num_formations': len(FORMATION_TO_IDX)
}

# --- Model Yolu ---
# Eğitilmiş modelinizin '.pth' veya '.pt' dosyasının doğru yolunu belirtin
DEFAULT_MODEL_PATH = "trained_models/hft_balanced_run_20250414_092547/best_model.pth" # Kendi yolunuzla değiştirin!

# --- Global Değişkenler ---
model: HierarchicalFormationTransformer = None # Model objesini tutacak
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Cihazı belirle

# --- Modeli Yükleme Fonksiyonu ---
def load_global_model(model_path: str):
    """
    Modeli oluşturur, belirtilen yoldan ağırlıkları yükler ve global değişkene atar.
    """
    global model
    print(f"Kullanılan cihaz: {device}")
    print("Model mimarisi oluşturuluyor...")

    # Map'lerden gerçek boyutları hesapla
    actual_class_vocab_size = len(CLASS_TO_IDX)
    actual_num_formations = len(FORMATION_TO_IDX)
    if actual_class_vocab_size == 0 or actual_num_formations == 0:
         print("HATA: CLASS_TO_IDX veya FORMATION_TO_IDX boş olamaz!")
         return

    try:
        # MODEL_CONFIG kullanarak modeli yarat
        model = HierarchicalFormationTransformer(
            num_formations=8,
            class_vocab_size=2,
            class_embed_dim=MODEL_CONFIG['class_embed_dim'],
            direction_dim=MODEL_CONFIG['direction_dim'],
            coord_dim=MODEL_CONFIG['coord_dim'],
            stage_dims=MODEL_CONFIG['stage_dims'],
            num_heads=MODEL_CONFIG['num_heads'],
            num_layers=MODEL_CONFIG['num_layers'],
            dropout_stages=MODEL_CONFIG['dropout_stages'],
            use_adapter=MODEL_CONFIG['use_adapter'],
            adapter_dim=MODEL_CONFIG['adapter_dim'],
            pos_type=MODEL_CONFIG['pos_type'],
            max_len=MODEL_CONFIG['max_len'],
            ffn_ratio=MODEL_CONFIG['ffn_ratio']
        )
        # Modeli uygun cihaza taşı (GPU veya CPU)
        model.to(device)

        # Model dosyasının varlığını kontrol et
        if not os.path.exists(model_path):
            print(f"HATA: Model dosyası bulunamadı: {model_path}")
            print(f"Lütfen DEFAULT_MODEL_PATH değişkenini doğru ayarlayın.")
            model = None # Hata durumunu belirt
            return

        # Model ağırlıklarını yükle
        print(f"Model ağırlıkları yükleniyor: {model_path}")
        # model_tools.py içindeki load_model fonksiyonunu çağır
        # Bu fonksiyonun modeli ve yolu alıp state_dict'i yüklediğini varsayıyoruz
        load_model(model_path, model) # Cihaz gerekirse buraya da eklenebilir
        model.eval() # Modeli tahmin/değerlendirme moduna al (Dropout vb. katmanları kapatır)
        print(f"Model başarıyla yüklendi '{model_path}' -> {device} ve eval() moduna alındı.")

    except FileNotFoundError:
        print(f"HATA: Model dosyası bulunamadı: {model_path}")
        model = None
    except ImportError:
         print(f"HATA: Model sınıfı veya yükleme fonksiyonu import edilemedi.")
         model = None
    except Exception as e:
        print(f"Model oluşturma/yükleme sırasında beklenmedik bir hata oluştu: {e}")
        print("Model konfigürasyonu (MODEL_CONFIG) ile yüklenen dosyanın uyumlu olduğundan emin olun.")
        model = None

# --- Tek Formasyon Tahmin Fonksiyonu ---
def predict_single_formation_from_data(data: Dict[str, Any]) -> int:
    """
    Flask isteğinden gelen JSON verisiyle tek bir formasyon tahmini yapar.
    Gerekli anahtarlar: 'coordinates', 'classes', 'directions'.
    """
    if model is None:
        raise RuntimeError("Model sunucuda yüklenmedi veya yüklenemedi.")

    try:
        # Gerekli verileri al
        coords = data['coordinates']
        class_names = data.get('classes', []) # 'classes' anahtarı yoksa boş liste
        directions = data['directions']

        # Sınıf isimlerini indekslere çevir
        class_tokens = [CLASS_TO_IDX.get(c, 0) for c in class_names] # Bilinmeyen sınıfa 0 ata

        # Temel Girdi Doğrulamaları
        if not isinstance(coords, list) or not isinstance(class_tokens, list) or not isinstance(directions, list):
            raise ValueError("Girdiler liste formatında olmalı (coordinates, class_tokens, directions).")
        if not coords:
             raise ValueError("Koordinat listesi boş olamaz.")
        if not (len(coords) == len(class_tokens) == len(directions)):
            raise ValueError(f"Girdi listelerinin uzunlukları eşleşmiyor: "
                             f"Coords({len(coords)}), Classes({len(class_tokens)}), Dirs({len(directions)})")
        if len(coords) > MODEL_CONFIG['max_len']:
             # Uzun sekanslar için uyarı ver (kırpma yapmıyoruz, modelin handle etmesi bekleniyor)
             print(f"Uyarı: Girdi uzunluğu ({len(coords)}) modelin max_len ({MODEL_CONFIG['max_len']}) değerinden büyük.")

        # Verileri PyTorch Tensor'larına çevir ve cihaza gönder
        # Unsqueeze(0) ile batch boyutu ekleniyor (batch_size=1)
        coords_t = torch.tensor(coords, dtype=torch.float32).unsqueeze(0).to(device)  # Shape: [1, N, coord_dim]
        class_tokens_t = torch.tensor(class_tokens, dtype=torch.long).unsqueeze(0).to(device)  # Shape: [1, N]

        # Yön verisinin boyutunu kontrol et (direction_dim'e göre)
        if MODEL_CONFIG['direction_dim'] == 1:
            # Shape: [N] -> [N, 1] -> [1, N, 1]
            directions_t = torch.tensor(directions, dtype=torch.float32).unsqueeze(-1).unsqueeze(0).to(device)
        elif MODEL_CONFIG['direction_dim'] == 2:
            # Shape: [N, 2] -> [1, N, 2]
            directions_t = torch.tensor(directions, dtype=torch.float32).unsqueeze(0).to(device)
            # Gelen verinin [N, 2] formatında olduğundan emin olun
            if directions_t.shape[-1] != 2:
                raise ValueError(f"direction_dim=2 ise yön verisi [N, 2] boyutunda olmalı, gelen: {directions_t.shape}")
        else:
            raise ValueError(f"Desteklenmeyen direction_dim: {MODEL_CONFIG['direction_dim']}")

        # Model ile tahmin yap (gradyan hesaplaması kapalı)
        with torch.no_grad():
            output_logits = model(coords=coords_t,
                                  class_tokens=class_tokens_t,
                                  directions=directions_t,
                                  key_padding_mask=None) # Tek örnek için padding maskesi None
            # En yüksek olasılıklı sınıfın indeksini al
            prediction_idx = torch.argmax(output_logits, dim=-1).item()

        return prediction_idx

    except KeyError as e:
        raise ValueError(f"JSON girdisinde beklenen anahtar bulunamadı: {e}")
    except TypeError as e:
        raise ValueError(f"Girdi verisinde tip hatası: {e}")
    except Exception as e:
        # Diğer beklenmedik hataları da yakala
        raise RuntimeError(f"Tahmin sırasında genel hata: {e}")


# --- Flask Uygulamasını Oluşturma ---
# static_folder ve template_folder yollarının doğru olduğundan emin olun
# Proje yapınız:
# /your_project_folder
#   ├── server.py
#   ├── model.py
#   ├── model_tools.py
#   ├── trained_models/ (içinde .pth dosyası)
#   ├── templates/
#   │   └── index.html
#   └── static/
#       ├── script.js
#       └── style.css
app = Flask(__name__, static_folder='static', template_folder='templates')

# --- CORS Yapılandırması ---
# Frontend'in çalıştığı kaynağa (örn: VS Code Live Server port 5500)
# backend API'sine (/predict) erişim izni vermek için.
# Eğer frontend ve backend aynı kaynaktan (örn: sadece Flask ile 5000 portundan)
# sunuluyorsa, CORS yapılandırmasına GEREK YOKTUR.
# Bu ayar, frontend'in http://127.0.0.1:5500 adresinden çalıştığı varsayımıyla yapılmıştır.
print("CORS yapılandırılıyor: '/predict' endpoint'ine 'http://127.0.0.1:5500' kaynağından erişime izin veriliyor.")
CORS(app, resources={r"/predict": {"origins": "http://127.0.0.1:5500"}})
# Dikkat: Production ortamında "origins" kısmını daha kısıtlayıcı yapın veya '*' kullanmaktan kaçının.
# Eğer frontend ve backend aynı Flask sunucusundan sunuluyorsa (Önerilen Yöntem),
# yukarıdaki CORS satırını silebilir veya yorum satırı yapabilirsiniz.


# --- Flask Rotaları (URL Yolları) ---

@app.route('/')
def index():
    """Ana sayfayı (index.html) sunar."""
    # templates klasöründeki index.html dosyasını render eder
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Statik dosyaları (CSS, JS) sunar."""
    # static klasöründen dosya gönderir
    return send_from_directory(app.static_folder, filename)

@app.route('/predict', methods=['POST'])
def predict_api():
    """
    Frontend'den gelen formasyon verisiyle tahmin yapar ve sonucu JSON olarak döner.
    Sadece POST isteklerini kabul eder.
    """
    # Modelin yüklenip yüklenmediğini kontrol et
    if model is None:
        # 503 Service Unavailable: Sunucu isteği işleyemiyor (model yok)
        return jsonify({"error": "Model sunucuda hazır değil veya yüklenemedi"}), 503

    # Gelen isteğin JSON formatında olup olmadığını kontrol et
    if not request.is_json:
        # 400 Bad Request: İstemci hatası (yanlış format)
        return jsonify({"error": "İstek JSON formatında olmalı"}), 400

    # İstekten JSON verisini al
    data = request.get_json()

    try:
        # Tahmin fonksiyonunu çağır
        prediction_idx = predict_single_formation_from_data(data)
        # Tahmin edilen indeksi formasyon ismine çevir
        predicted_formation_name = IDX_TO_FORMATION.get(prediction_idx, "Bilinmeyen Formasyon")

        # Başarılı yanıtı JSON olarak gönder
        # 200 OK (Varsayılan)
        return jsonify({
            "prediction": predicted_formation_name,
            # İsteğe bağlı: Orijinal formasyon ismini de geri gönderebiliriz (frontend'de karşılaştırma için)
            # "original": data.get("formation", "N/A")
            })

    except ValueError as e: # Girdi verisiyle ilgili beklenen hatalar
        # 400 Bad Request: İstemci hatası (geçersiz girdi)
        return jsonify({"error": f"Geçersiz Girdi: {e}"}), 400
    except RuntimeError as e: # Model veya tahmin sırasındaki beklenmedik hatalar
        # 500 Internal Server Error: Sunucu hatası
         return jsonify({"error": f"Tahmin Hatası: {e}"}), 500
    except Exception as e: # Diğer tüm beklenmedik hatalar
        print(f"Beklenmedik Sunucu Hatası: {e}") # Hata loglaması için
        # 500 Internal Server Error
        return jsonify({"error": "Sunucuda beklenmedik bir hata oluştu"}), 500


# --- Sunucuyu Başlatma Bloğu ---
if __name__ == '__main__':
    # Sunucu başlamadan ÖNCE modeli yükle
    load_global_model(DEFAULT_MODEL_PATH)

    # Flask geliştirme sunucusunu başlat
    # debug=True: Kod değişikliklerinde sunucuyu otomatik yeniden başlatır ve daha detaylı hata mesajları verir.
    #             Production ortamında debug=False kullanılmalıdır.
    # host='127.0.0.1': Sunucunun sadece yerel makineden erişilebilir olmasını sağlar.
    #                 Ağdaki diğer cihazların erişmesi için host='0.0.0.0' kullanın.
    # port=5000: Sunucunun çalışacağı port numarası. Başka bir uygulama kullanıyorsa değiştirin.
    print("Flask sunucusu başlatılıyor...")
    app.run(debug=True, host='127.0.0.1', port=5000)