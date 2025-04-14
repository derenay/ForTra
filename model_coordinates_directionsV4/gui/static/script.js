// static/script.js
// Son Güncelleme: 2025-04-11 (Modernizasyon ve Tank İkonu)
// Açıklama: JSON çıktısındaki koordinatlar Matematik Standardına (Sol Alt Orijin, +Y Yukarı),
//          Açılar ve görselleştirme Matematik Standardına (0=Sağ, 90=Yukarı) göre ayarlandı.
//          Tanklar artık ikon olarak çiziliyor.

const canvas = document.getElementById('formationCanvas');
const ctx = canvas.getContext('2d');
const formationNameInput = document.getElementById('formationName');
const saveFormationButton = document.getElementById('saveFormation');
const clearCurrentButton = document.getElementById('clearCurrent');
const exportJsonButton = document.getElementById('exportJson');
const infoArea = document.getElementById('infoArea');
const savedList = document.getElementById('savedList');
const startAnimationButton = document.getElementById('startAnimation');
const stopAnimationButton = document.getElementById('stopAnimation');
startAnimationButton.addEventListener('click', startTankAnimation);
stopAnimationButton.addEventListener('click', stopTankAnimation);


let currentTanks = []; // Mevcut formasyondaki tankları tutar: { x, y (Canvas std), dir_normalized (Math std), temp }
let allFormations = []; // Kaydedilmiş tüm formasyonları tutar (Koordinatlar Math std olacak)
let isPlacingTank = true; // Durum: Tank mı yerleştiriliyor, yön mü belirleniyor?
let animationIntervalId; // Animasyon döngüsü için ID
let isAnimating = false;   // Animasyonun çalışıp çalışmadığını belirtir
const animationSpeed = 0.005; // Tankların hareket hızı (normalize edilmiş birim/frame)
// Tank Görsel Ayarları (Pixel Cinsinden)
const TANK_WIDTH_PX = 16;
const TANK_HEIGHT_PX = 20; // Genişliğinden biraz daha uzun
const TURRET_RADIUS_PX = 5;
const BARREL_LENGTH_PX = 15;
const BARREL_WIDTH_PX = 3;

// Eğitim verisindeki FORMATION_TO_IDX anahtarlarıyla eşleşmeli
const FORMATION_NAMES = ["Line", "Wedge", "Vee", "Herringbone", "Coil", "Staggered Column", "Column", "Echelon"];

//--- Olay Dinleyiciler ---
canvas.addEventListener('click', handleCanvasClick);
saveFormationButton.addEventListener('click', saveCurrentFormation);
clearCurrentButton.addEventListener('click', clearCurrentFormation);
exportJsonButton.addEventListener('click', exportAllFormationsToJson);
savedList.addEventListener('click', handleSavedListClick); // Test butonları için birleşik dinleyici

//--- Fonksiyonlar ---

/**
 * Sayfa yüklendiğinde formasyon seçme kutusunu doldurur.
 */
function populateFormationSelect() {
    const selectElement = document.getElementById('formationName');
    selectElement.innerHTML = '<option value="" disabled selected>Formasyon Seçin...</option>';
    FORMATION_NAMES.forEach(name => {
        const option = document.createElement('option');
        option.value = name;
        option.textContent = name;
        selectElement.appendChild(option);
    });
     // Seçim yapıldığında placeholder'ı gizle (isteğe bağlı)
    selectElement.addEventListener('change', () => {
        if (selectElement.value) {
            selectElement.querySelector('option[disabled]').style.display = 'none';
        }
    });
}



let zoomLevel = 1.0;
const minZoom = 0.5;
const zoomSpeed = 0.005; // Zoom hızı
const collisionThresholdSq = Math.pow(TANK_WIDTH_PX * 1.5, 2); // Çarpışma eşiği (kareli mesafe)

function drawCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Tüm tankların merkez noktasını bul
    let totalX = 0;
    let totalY = 0;
    currentTanks.forEach(tank => {
        totalX += tank.x;
        totalY += tank.y;
    });
    const avgX = currentTanks.length > 0 ? totalX / currentTanks.length : 0.5;
    const avgY = currentTanks.length > 0 ? totalY / currentTanks.length : 0.5;

    ctx.save();

    // Zoom uygula (merkeze göre)
    const centerX = avgX * canvas.width;
    const centerY = avgY * canvas.height;
    ctx.translate(centerX, centerY);
    ctx.scale(zoomLevel, zoomLevel);
    ctx.translate(-centerX, -centerY);

    drawGrid();

    currentTanks.forEach((tank, index) => {
        const px = tank.x * canvas.width;
        const py = tank.y * canvas.height;
        const dirNormalized = tank.dir_normalized;
        const dirRadiansMath = dirNormalized * 2 * Math.PI;
        const dirRadiansForCanvasRotation = -dirRadiansMath;

        ctx.save();
        ctx.translate(px, py);
        ctx.rotate(dirRadiansForCanvasRotation);

        ctx.fillStyle = tank.temp ? '#FFA500' : '#007bff';
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;

        ctx.beginPath();
        ctx.rect(-TANK_WIDTH_PX / 2, -TANK_HEIGHT_PX / 2, TANK_WIDTH_PX, TANK_HEIGHT_PX);
        ctx.fill();
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(0, 0, TURRET_RADIUS_PX, 0, 2 * Math.PI);
        ctx.fillStyle = tank.temp ? '#FFC107' : '#0056b3';
        ctx.fill();
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(TURRET_RADIUS_PX + BARREL_LENGTH_PX, 0);
        ctx.strokeStyle = '#333';
        ctx.lineWidth = BARREL_WIDTH_PX;
        ctx.stroke();

        ctx.restore();

        ctx.fillStyle = '#333';
        ctx.font = 'bold 11px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(index + 1, px, py - TANK_HEIGHT_PX / 2 - 5);
    });

    ctx.restore();

    // Basit zoom ayarı (tüm tankların birbirine olan mesafesine göre)
    if (isAnimating && currentTanks.length > 1) {
        let minDistanceSq = Infinity;
        for (let i = 0; i < currentTanks.length; i++) {
            for (let j = i + 1; j < currentTanks.length; j++) {
                const dx = (currentTanks[i].x - currentTanks[j].x) * canvas.width;
                const dy = (currentTanks[i].y - currentTanks[j].y) * canvas.height;
                const distSq = dx * dx + dy * dy;
                minDistanceSq = Math.min(minDistanceSq, distSq);
            }
        }

        if (minDistanceSq < collisionThresholdSq && zoomLevel > minZoom) {
            zoomLevel -= zoomSpeed;
        } else if (zoomLevel < 1.0 && minDistanceSq > collisionThresholdSq * 2) { // Biraz uzaklaştıktan sonra geri dön
            zoomLevel += zoomSpeed * 0.5;
            zoomLevel = Math.min(1.0, zoomLevel);
        }
        zoomLevel = Math.max(minZoom, Math.min(1.0, zoomLevel)); // Zoom sınırlarını koru
    } else if (!isAnimating && zoomLevel < 1.0) {
        zoomLevel += zoomSpeed * 0.5;
        zoomLevel = Math.min(1.0, zoomLevel);
    }
}


function handleCanvasClick(event) {
    const rect = canvas.getBoundingClientRect();
    const canvasX = event.clientX - rect.left;
    const canvasY = event.clientY - rect.top;

    // Canvas sınırları dışındaki tıklamaları yok say
    if (canvasX < 0 || canvasX > canvas.width || canvasY < 0 || canvasY > canvas.height) {
        return;
    }

    // Tıklama konumunu Canvas standardına göre normalize et ([0,1], sol üst orijin)
    const normX = canvasX / canvas.width;
    const normY = canvasY / canvas.height;

    if (isPlacingTank) {
        // Yeni tank yerleştir (koordinatlar Canvas std, yön Math std olacak)
        currentTanks.push({
            x: normX, // Canvas standardı X
            y: normY, // Canvas standardı Y
            dir_normalized: 0.0, // Varsayılan yön (Matematik Standardında 0 = Sağ)
            temp: true // Henüz yönü kesinleşmedi
        });
        isPlacingTank = false; // Sonraki tıklama yönü belirleyecek
        updateInfoArea("Tank yerleştirildi. Yönü belirlemek için tekrar tıklayın.");
    } else {
        // Son eklenen tankın yönünü belirle
        if (currentTanks.length > 0) {
            const lastTank = currentTanks[currentTanks.length - 1];
            if (lastTank.temp) { // Sadece yönü ayarlanmamışsa
                // Tankın piksel konumunu (Canvas std) hesapla
                const tankPx = lastTank.x * canvas.width;
                const tankPy = lastTank.y * canvas.height;

                // Vektörü hesapla (Canvas std: +Y aşağı)
                const dx = canvasX - tankPx;
                const dy = canvasY - tankPy;

                // Açıyı hesapla (Canvas std: Aşağı=90)
                const angleRadiansOriginal = Math.atan2(dy, dx);

                // Açıyı Matematik Standardına çevir (Y eksenini ters çevir: Yukarı=90)
                const angleRadiansCorrected = -angleRadiansOriginal;

                // Dereceye çevir (Matematik Standardına göre)
                const angleDegrees = (angleRadiansCorrected * 180 / Math.PI + 360) % 360;

                // Matematik Standardındaki normalize açıyı kaydet
                lastTank.dir_normalized = normalizeAngleDegrees(angleDegrees);
                lastTank.temp = false; // Yön belirlendi
                isPlacingTank = true; // Bir sonraki tıklama yeni tank yerleştirecek
                updateInfoArea(`Son tankın yönü ${angleDegrees.toFixed(1)}° olarak ayarlandı (0°=Sağ, 90°=Yukarı). Yeni tank yerleştirmek için tıklayın.`);
            } else {
                 updateInfoArea("Yeni tank yerleştirmek için tıklayın.");
                 isPlacingTank = true; // Beklenmedik durumlar için durumu sıfırla
            }
        }
    }
    drawCanvas(); // Her tıklamadan sonra canvas'ı yeniden çiz
}

/**
 * Canvas üzerine yardımcı bir ızgara çizer.
 */
function drawGrid() {
    ctx.strokeStyle = '#e9ecef'; // Izgara rengi (daha açık)
    ctx.lineWidth = 1;
    const step = 50; // Izgara aralığı (pixel)

    // Dikey çizgiler
    for (let x = 0; x <= canvas.width; x += step) {
        ctx.beginPath();
        ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height);
        ctx.stroke();
    }
    // Yatay çizgiler
    for (let y = 0; y <= canvas.height; y += step) {
        ctx.beginPath();
        ctx.moveTo(0, y); ctx.lineTo(canvas.width, y);
        ctx.stroke();
    }
     // Merkez çizgileri (isteğe bağlı, biraz daha belirgin)
     ctx.strokeStyle = '#ced4da';
     ctx.lineWidth = 1;
     ctx.beginPath(); ctx.moveTo(canvas.width/2, 0); ctx.lineTo(canvas.width/2, canvas.height); ctx.stroke();
     ctx.beginPath(); ctx.moveTo(0, canvas.height/2); ctx.lineTo(canvas.width, canvas.height/2); ctx.stroke();
}

/**
 * Bilgi alanını günceller.
 */
function updateInfoArea(message = "") {
    let html = "<h3>Durum</h3>";
     if (message) {
        html += `<p>${message}</p>`;
    }

    html += "<h4>Mevcut Tanklar:</h4>";
    if (currentTanks.length === 0) {
        html += "<p>Henüz tank eklenmedi.</p>";
    } else {
        html += "<ul>";
        currentTanks.forEach((tank, index) => {
            // Gösterilen derece Matematik Standardı (0=Sağ, 90=Yukarı)
            const displayDegrees = (tank.dir_normalized * 360).toFixed(1);
            // Durum için ikon veya metin (CSS sınıfları ile stil verilebilir)
            const statusClass = tank.temp ? 'status-pending' : 'status-complete';
            const statusIcon = tank.temp ? '🟠 Yön bekleniyor' : '🟢 Tamamlandı'; // Basit ikonlar
            // Info alanında Canvas koordinatlarını ve Math yönünü gösterelim
            html += `<li class="${statusClass}">Tank ${index + 1}: Koordinat (${tank.x.toFixed(3)}, ${tank.y.toFixed(3)}), Yön: ${displayDegrees}° - ${statusIcon}</li>`;
        });
        html += "</ul>";
    }
    // Sonraki eylem için ipucu
    if (!isPlacingTank && currentTanks.length > 0 && currentTanks[currentTanks.length - 1].temp){
         html += "<p style='color:orange; font-weight:bold;'>➡️ Yönü belirlemek için tekrar tıklayın.</p>";
    } else {
         html += "<p>🖱️ Yeni tank yerleştirmek için tıklayın.</p>";
    }
     infoArea.innerHTML = html;
}

/**
 * Derece cinsinden açıyı [0, 1) aralığına normalize eder.
 */
function normalizeAngleDegrees(degrees) {
    const normalized = (degrees % 360) / 360.0;
    return normalized < 0 ? normalized + 1 : normalized;
}

/**
 * Mevcut tank düzenini bir formasyon olarak kaydeder.
 * Koordinatları Matematik Standardına çevirir.
 */
function saveCurrentFormation() {
    const name = formationNameInput.value;
    if (!name) {
        alert("Lütfen formasyon için listeden bir isim seçin.");
        formationNameInput.focus(); // Seçim kutusuna odaklan
        return;
    }
    if (currentTanks.length === 0) {
        alert("Kaydedilecek tank bulunmuyor.");
        return;
    }
    const pendingTank = currentTanks.find(tank => tank.temp);
    if (pendingTank) {
        alert("Lütfen kaydetmeden önce tüm tankların yönünü ayarlayın.\n(Turuncu renkli tankın yönü eksik)");
        return;
    }
    const minTankCount = 2
    if (currentTanks.length < minTankCount){
        alert(`Lütfen en az ${minTankCount} kadar Tank yerleştirin`)
        return
    }

    // Canvas standardı koordinatlarını Matematik standardına çevir
    const coordinates_math_standard = currentTanks.map(tank => [
        round(tank.x, 5),        // X aynı kalır
        round(1.0 - tank.y, 5)   // Y'yi dönüştür: 1 - y
    ]);
    // Yönleri al (zaten Matematik Standardında)
    const directions_math_standard = currentTanks.map(tank => round(tank.dir_normalized, 5));
    // Sınıfları al (şimdilik sabit)
    const classes = Array(currentTanks.length).fill("tank");

    const newFormation = {
        coordinates: coordinates_math_standard,
        classes: classes,
        formation: name,
        directions: directions_math_standard
    };

    allFormations.push(newFormation);
    const newIndex = allFormations.length - 1;
    updateSavedList(name, currentTanks.length, newIndex); // Listeyi güncelle

    // Arayüzü temizle ve bilgilendir
    // formationNameInput.value = ""; // Seçili kalsın, tekrar aynı formasyon eklenebilir
    formationNameInput.selectedIndex = 0; // Placeholder'a geri dön
    formationNameInput.querySelector('option[disabled]').style.display = ''; // Placeholder'ı göster

    const savedTanksCount = currentTanks.length; // Sayıyı kaydetmeden önce al
    clearCurrentFormation(false); // Canvas'ı temizle (mesaj gösterme)
    updateInfoArea(`✅ '${name}' formasyonu ${savedTanksCount} tank ile kaydedildi. Yeni formasyon oluşturabilirsiniz.`);
    isPlacingTank = true; // Yeni formasyona başla durumu
}

/**
 * Kaydedilen formasyonlar listesine yeni bir öğe ekler (Test butonu ile).
 */
function updateSavedList(formationName, tankCount, formationIndex){
    const listItem = document.createElement('li');
    // Daha iyi erişilebilirlik için butonlara ID ekleyelim
    listItem.innerHTML = `
        <span>${formationName} (${tankCount} tank)</span>
        <div>
            <span class="prediction-result" id="pred-result-${formationIndex}"></span>
            <button class="test-button" id="test-btn-${formationIndex}" data-formation-index="${formationIndex}">Test Et</button>
        </div>
    `;
    savedList.appendChild(listItem);
     // Yeni eklenen öğeyi görünür yap (eğer scroll varsa)
    listItem.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Canvas üzerindeki mevcut tankları ve çizimi temizler.
 */
function clearCurrentFormation(showAlert = true) {
    const tankCount = currentTanks.length;
    currentTanks = [];
    isPlacingTank = true;
    drawCanvas(); // Boş canvas'ı çiz
    if (showAlert && tankCount > 0) {
        updateInfoArea("🗑️ Mevcut tanklar temizlendi. Yeni formasyona başlayabilirsiniz.");
    } else {
         updateInfoArea(); // Sadece durumu güncelle (genellikle kaydetme sonrası)
    }
}

/**
 * Kaydedilmiş tüm formasyonları bir JSON dosyası olarak indirir.
 */
function exportAllFormationsToJson() {
    if (allFormations.length === 0) {
        alert("Dışa aktarılacak kaydedilmiş formasyon bulunmuyor.");
        return;
    }

    const jsonData = JSON.stringify(allFormations, null, 2); // null, 2 ile güzel formatlama
    const blob = new Blob([jsonData], { type: 'application/json;charset=utf-8' }); // UTF-8 karakterler için
    const url = URL.createObjectURL(blob);

    const link = document.createElement('a');
    link.setAttribute('href', url);
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    link.setAttribute('download', `tank_formations_${timestamp}.json`); // Daha açıklayıcı dosya adı
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    updateInfoArea(`💾 Toplam ${allFormations.length} formasyon JSON dosyası olarak indirildi.`);
}

/**
 * Sayıyı belirli bir ondalık basamağa yuvarlar.
 */
function round(value, decimals) {
  return Number(Math.round(value + 'e' + decimals) + 'e-' + decimals);
}

/**
 * Kaydedilenler listesindeki tıklama olaylarını yönetir (Test butonları).
 */
function handleSavedListClick(event) {
    if (event.target.tagName === 'BUTTON' && event.target.classList.contains('test-button')) {
        const button = event.target;
        const formationIndex = parseInt(button.getAttribute('data-formation-index'));
         if (!isNaN(formationIndex)) {
            handleTestButtonClick(formationIndex, button); // Async fonksiyonu çağır
         } else {
             console.error("Geçersiz formasyon index:", button.getAttribute('data-formation-index'));
             alert("Test butonu için geçersiz formasyon index'i!");
         }
    }
}



function animateTanks() {
    if (!isAnimating) {
        return;
    }

    // Her tankın yeni konumunu hesapla
    currentTanks.forEach(tank => {
        // Normalize edilmiş açıyı radyana çevir (Matematik standardı)
        const angleRadians = tank.dir_normalized * 2 * Math.PI;
        // Yön vektörünü hesapla
        const directionX = Math.cos(angleRadians);
        const directionY = Math.sin(angleRadians);
        // Yeni konumu hesapla (Canvas standardında kalmaya devam ediyoruz)
        tank.x += directionX * animationSpeed;
        tank.y -= directionY * animationSpeed; // Canvas Y ekseni ters olduğu için çıkarma

        // Sınır kontrolü (isteğe bağlı, tankları ekranda tutmak için)
        if (tank.x < 0 || tank.x > 1 || tank.y < 0 || tank.y > 1) {
            // Tank sınırı aştıysa ne yapılacağı (örneğin durdurma, ters çevirme vb.)
            console.log(`Tank sınırları aştı: ${tank.x}, ${tank.y}`);
            // İsteğe bağlı: tank.x = Math.max(0, Math.min(1, tank.x));
            // İsteğe bağlı: tank.y = Math.max(0, Math.min(1, tank.y));
        }
    });

    // Canvas'ı yeniden çiz
    drawCanvas();

    // Bir sonraki animasyon karesini iste
    animationIntervalId = requestAnimationFrame(animateTanks);
}


function startTankAnimation() {
    if (currentTanks.length === 0) {
        alert("Lütfen önce tank yerleştirin.");
        return;
    }
    if (currentTanks.some(tank => tank.temp)) {
        alert("Lütfen tüm tankların yönünü belirleyin.");
        return;
    }
    isAnimating = true;
    startAnimationButton.disabled = true;
    stopAnimationButton.disabled = false;
    animateTanks();
}


function stopTankAnimation() {
    isAnimating = false;
    startAnimationButton.disabled = false;
    stopAnimationButton.disabled = true;
    if (animationIntervalId) {
        cancelAnimationFrame(animationIntervalId);
        animationIntervalId = null;
    }
}



/**
 * "Test Et" butonuna tıklandığında backend API'sine istek gönderir.
 */
async function handleTestButtonClick(formationIndex, buttonElement) {
    if (formationIndex < 0 || formationIndex >= allFormations.length) {
        console.error("Geçersiz formasyon index:", formationIndex);
        return;
    }
    const formationData = allFormations[formationIndex];
    const resultSpan = document.getElementById(`pred-result-${formationIndex}`);
    if (!resultSpan) {
        console.error("Sonuç span elementi bulunamadı:", `pred-result-${formationIndex}`);
        return;
    }

    buttonElement.disabled = true;
    buttonElement.textContent = "Test Ediliyor..."; // Buton metnini güncelle
    resultSpan.textContent = "⏳"; // Bekleme ikonu
    resultSpan.className = 'prediction-result status-testing'; // CSS sınıfı ile stil ver

    try {
        const payload = {
            coordinates: formationData.coordinates,
            classes: formationData.classes,
            directions: formationData.directions
        };

        // API Endpoint - Gerekirse değiştirin
        const apiUrl = 'http://127.0.0.1:5000/predict';

        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload),
            // Timeout eklemek iyi bir pratik olabilir (örneğin 10 saniye)
            // signal: AbortSignal.timeout(10000)
        });

        if (!response.ok) {
            let errorMsg = `HTTP Hatası! Kod: ${response.status}`;
            try {
                const errorData = await response.json();
                errorMsg = errorData.error || errorMsg; // Backend'den gelen hata mesajını kullan
            } catch (e) { /* JSON parse hatası olursa orijinal mesajı kullan */ }
            throw new Error(errorMsg);
        }

        const result = await response.json();
        resultSpan.textContent = `✔️ Tahmin: ${result.prediction}`;
        resultSpan.className = 'prediction-result status-success'; // Başarı stili

    } catch (error) {
        console.error('Tahmin isteği hatası:', error);
        resultSpan.textContent = `❌ Hata: ${error.message}`;
        resultSpan.className = 'prediction-result status-error'; // Hata stili
    } finally {
        // Butonu tekrar etkinleştir ve orijinal metne döndür
        buttonElement.disabled = false;
        buttonElement.textContent = "Test Et";
    }
}

// --- Sayfa Yüklendiğinde Çalışacak İlk Kodlar ---
document.addEventListener('DOMContentLoaded', () => {
    populateFormationSelect(); // Açılır listeyi doldur
    drawCanvas(); // Canvas'ı ilk kez çiz (ızgara vb. için)
    updateInfoArea("🚀 Başlamak için canvas'a tıklayarak ilk tankı yerleştirin."); // Başlangıç mesajı
});