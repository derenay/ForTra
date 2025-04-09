// static/script.js
// Son Güncelleme: 2025-04-09
// Açıklama: JSON çıktısındaki koordinatlar Matematik Standardına (Sol Alt Orijin, +Y Yukarı),
//          Açılar ve görselleştirme Matematik Standardına (0=Sağ, 90=Yukarı) göre ayarlandı.

const canvas = document.getElementById('formationCanvas');
const ctx = canvas.getContext('2d');
const formationNameInput = document.getElementById('formationName');
const saveFormationButton = document.getElementById('saveFormation');
const clearCurrentButton = document.getElementById('clearCurrent');
const exportJsonButton = document.getElementById('exportJson');
const infoArea = document.getElementById('infoArea');
const savedList = document.getElementById('savedList');


let currentTanks = []; // Mevcut formasyondaki tankları tutar: { x, y (Canvas std), dir_normalized (Math std), temp }
let allFormations = []; // Kaydedilmiş tüm formasyonları tutar (Koordinatlar Math std olacak)
let isPlacingTank = true; // Durum: Tank mı yerleştiriliyor, yön mü belirleniyor?
const TANK_RADIUS_PX = 8; // Tank yarıçapı (pixel)
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
}

/**
 * Canvas üzerindeki tıklamaları yönetir: Tank yerleştirir veya yön belirler.
 */


// Tankları çiz
function drawCanvas() {
    // Canvas'ı temizle
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // İsteğe bağlı: Izgara çizimi
    drawGrid();

    // Tankları çiz
    currentTanks.forEach((tank, index) => {
        // Tank konumunu piksele çevir (Canvas std)
        const px = tank.x * canvas.width;
        const py = tank.y * canvas.height;

        // Kaydedilmiş normalize yönü al (bu Matematik Standardında: Yukarı=0.25)
        const dirNormalized = tank.dir_normalized;
        // Normalizasyondan radyana çevir
        const dirRadiansStored = dirNormalized * 2 * Math.PI;

        // *** GÖRSEL DÜZELTME: AÇIYI TEKRAR TERS ÇEVİR ***
        // Saklanan açı Math standardında (Yukarı=PI/2). Canvas'ta okun YUKARI
        // görünmesi için, rotate fonksiyonuna açının NEGATİFİNİ vermeliyiz.
        // (Çünkü rotate(PI/2) oku AŞAĞI çizer, rotate(-PI/2) ise YUKARI çizer).
        const dirRadiansForCanvasRotation = -dirRadiansStored;
        // *** DÜZELTME SONU ***

        // Tank Gövdesi
        ctx.beginPath();
        ctx.arc(px, py, TANK_RADIUS_PX, 0, 2 * Math.PI);
        ctx.fillStyle = tank.temp ? 'orange' : '#007bff';
        ctx.fill();
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 1;
        ctx.stroke();

        // Yön Oku
        ctx.save();
        ctx.translate(px, py);
        // Görsel düzeltme için negatiflenmiş açıyı kullanarak döndür
        ctx.rotate(dirRadiansForCanvasRotation);

        // Oku +X yönünde çiz
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(TANK_RADIUS_PX + 5, 0);
        ctx.strokeStyle = 'red';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Ok ucu
        ctx.beginPath();
        ctx.moveTo(TANK_RADIUS_PX + 5, 0);
        ctx.lineTo(TANK_RADIUS_PX + 1, -3);
        ctx.lineTo(TANK_RADIUS_PX + 1, 3);
        ctx.closePath();
        ctx.fillStyle = 'red';
        ctx.fill();

        ctx.restore();

        // Tank numarası
            ctx.fillStyle = 'black';
            ctx.font = '10px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(index + 1, px, py - TANK_RADIUS_PX - 2);
    });
}

/**
 * Canvas üzerine yardımcı bir ızgara çizer.
 */
function drawGrid() {
    ctx.strokeStyle = '#e0e0e0'; // Izgara rengi
    ctx.lineWidth = 0.5;
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
     // Merkez çizgileri (isteğe bağlı, daha belirgin)
     ctx.strokeStyle = '#c0c0c0';
     ctx.lineWidth = 1;
     ctx.beginPath(); ctx.moveTo(canvas.width/2, 0); ctx.lineTo(canvas.width/2, canvas.height); ctx.stroke();
     ctx.beginPath(); ctx.moveTo(0, canvas.height/2); ctx.lineTo(canvas.width, canvas.height/2); ctx.stroke();
}


function updateInfoArea(message = "") {
    let html = "<h3>Durum</h3>";
     if (message) {
        html += `<p>${message}</p>`;
    }

    html += "<h4>Mevcut Tanklar (Canvas Koordinatları):</h4>"; // Koordinatların Canvas std olduğunu belirtelim
    if (currentTanks.length === 0) {
        html += "<p>Henüz tank eklenmedi.</p>";
    } else {
        html += "<ul>";
        currentTanks.forEach((tank, index) => {
            // Gösterilen derece artık Matematik Standardı (0=Sağ, 90=Yukarı)
            const displayDegrees = (tank.dir_normalized * 360).toFixed(1);
            const status = tank.temp ? "<i style='color:orange;'>(Yön bekleniyor)</i>" : "<i style='color:green;'>(Tamamlandı)</i>";
            // Info alanında Canvas koordinatlarını gösterelim (çünkü içsel olarak onlar kullanılıyor)
            html += `<li>Tank ${index + 1}: (${tank.x.toFixed(3)}, ${tank.y.toFixed(3)}), Yön: ${displayDegrees}° ${status}</li>`;
        });
        html += "</ul>";
    }
    // Sonraki eylem için ipucu
    if (!isPlacingTank && currentTanks.length > 0 && currentTanks[currentTanks.length -1].temp){
         html += "<p style='color:orange; font-weight:bold;'>Yönü belirlemek için tekrar tıklayın.</p>";
    } else {
         html += "<p>Yeni tank yerleştirmek için tıklayın.</p>";
    }
     infoArea.innerHTML = html;
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
 * Canvas üzerine mevcut tankları ve yön oklarını çizer.
 */

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
    // Select elementinden formasyon adını al
    const name = formationNameInput.value;
    if (!name) {
        alert("Lütfen formasyon için listeden bir isim seçin.");
        return;
    }
    // Tank olup olmadığını ve yönlerin ayarlanıp ayarlanmadığını kontrol et
    if (currentTanks.length === 0) {
        alert("Kaydedilecek tank bulunmuyor.");
        return;
    }
    const pendingTank = currentTanks.find(tank => tank.temp);
    if (pendingTank) {
        alert("Lütfen kaydetmeden önce tüm tankların yönünü ayarlayın.");
        return;
    }

    // --- KOORDİNAT DÖNÜŞÜMÜ (JSON için) ---
    // Canvas standardı koordinatlarını Matematik standardına çevir
    const coordinates_math_standard = currentTanks.map(tank => [
        round(tank.x, 4),        // X aynı kalır
        round(1.0 - tank.y, 4)   // Y'yi dönüştür: 1 - y
    ]);
    // --- DÖNÜŞÜM SONU ---

    // Yönleri al (bunlar zaten Matematik Standardında kaydedildi)
    const directions_math_standard = currentTanks.map(tank => round(tank.dir_normalized, 4));
    // Sınıfları al
    const classes = Array(currentTanks.length).fill("tank"); // Şimdilik sadece tank

    // Yeni formasyon objesini oluştur
    const newFormation = {
        coordinates: coordinates_math_standard, // Matematik std koordinatlar
        classes: classes,
        formation: name,
        directions: directions_math_standard // Matematik std yönler
    };

    // Kaydedilen formasyonlar listesine ekle
    allFormations.push(newFormation);
    const newIndex = allFormations.length - 1;
    updateSavedList(name, currentTanks.length, newIndex); // Listeyi güncelle

    // Arayüzü temizle ve bilgilendir
    formationNameInput.value = ""; // Select'i sıfırla
    clearCurrentFormation(false); // Canvas'ı temizle (mesaj gösterme)
    updateInfoArea(`'${name}' formasyonu ${currentTanks.length} tank ile kaydedildi. Yeni formasyon oluşturabilirsiniz.`);
    isPlacingTank = true; // Yeni formasyona başla durumu
}

/**
 * Kaydedilen formasyonlar listesine yeni bir öğe ekler (Test butonu ile).
 */
function updateSavedList(formationName, tankCount, formationIndex){
    const listItem = document.createElement('li');
    listItem.innerHTML = `
        <span>${formationName} (${tankCount} tank)</span>
        <button class="test-button" data-formation-index="${formationIndex}">Test Et</button>
        <span class="prediction-result" id="pred-result-${formationIndex}"></span>
    `;
    savedList.appendChild(listItem);
}

/**
 * Canvas üzerindeki mevcut tankları ve çizimi temizler.
 */
function clearCurrentFormation(showAlert = true) {
    currentTanks = []; // Mevcut tank listesini boşalt
    isPlacingTank = true; // Durumu sıfırla
    drawCanvas(); // Boş canvas'ı çiz
    if (showAlert) {
        updateInfoArea("Mevcut tanklar temizlendi. Yeni formasyona başlayabilirsiniz.");
    } else {
        updateInfoArea(); // Sadece durumu güncelle
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

    // allFormations listesini JSON string'ine çevir (okunabilir formatta)
    const jsonData = JSON.stringify(allFormations, null, 2);
    const blob = new Blob([jsonData], { type: 'application/json;charset=utf-8,' });
    const url = URL.createObjectURL(blob);

    const link = document.createElement('a');
    link.setAttribute('href', url);
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    link.setAttribute('download', `manual_formations_${timestamp}.json`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click(); // İndirmeyi tetikle
    document.body.removeChild(link); // Linki kaldır
    URL.revokeObjectURL(url); // Geçici URL'yi serbest bırak

    updateInfoArea(`Toplam ${allFormations.length} formasyon JSON dosyası olarak indirildi.`);
}

/**
 * Sayıyı belirli bir ondalık basamağa yuvarlar.
 */
function round(value, decimals) {
  return Number(Math.round(value + 'e' + decimals) + 'e-' + decimals);
}

/**
 * Kaydedilenler listesindeki tıklama olaylarını yönetir (özellikle Test butonları).
 */
function handleSavedListClick(event) {
    // Tıklanan eleman 'test-button' class'ına sahip bir BUTON mu?
    if (event.target.tagName === 'BUTTON' && event.target.classList.contains('test-button')) {
        const button = event.target;
        // Butonun data attribute'undan ilgili formasyonun index'ini al
        const formationIndex = parseInt(button.getAttribute('data-formation-index'));
         if (!isNaN(formationIndex)) {
            // Test işlemini başlatacak fonksiyonu çağır
            handleTestButtonClick(formationIndex, button);
         } else {
             console.error("Geçersiz formasyon index:", button.getAttribute('data-formation-index'));
             alert("Test butonu için geçersiz formasyon index'i!");
         }
    }
}

/**
 * "Test Et" butonuna tıklandığında backend API'sine istek gönderir.
 */
async function handleTestButtonClick(formationIndex, buttonElement) {
    // Index'in geçerli olduğundan emin ol
    if (formationIndex < 0 || formationIndex >= allFormations.length) {
        console.error("Geçersiz formasyon index:", formationIndex);
        return;
    }
    // İlgili formasyon verisini al (Bu veride koordinatlar zaten Math Std)
    const formationData = allFormations[formationIndex];
    // Sonucun gösterileceği span elementini ID ile bul
    const resultSpan = document.getElementById(`pred-result-${formationIndex}`);
    if (!resultSpan) {
        console.error("Sonuç span elementi bulunamadı:", `pred-result-${formationIndex}`);
        return;
    }

    // Butonu geçici olarak devre dışı bırak ve durumu güncelle
    buttonElement.disabled = true;
    resultSpan.textContent = " - Test ediliyor...";
    resultSpan.style.color = 'orange';

    try {
        // Backend'e gönderilecek veri (JSON'daki haliyle)
        const payload = {
            coordinates: formationData.coordinates, // Matematik Std Koordinatlar
            classes: formationData.classes,
            directions: formationData.directions   // Matematik Std Yönler
        };

        // Fetch API ile Flask sunucusundaki /predict adresine POST isteği gönder
        const response = await fetch('http://127.0.0.1:5000/predict', { // Backend adresini kontrol edin
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload),
        });

        // Yanıtı kontrol et
        if (!response.ok) {
            let errorMsg = `HTTP Hatası! Durum: ${response.status}`;
            try {
                const errorData = await response.json();
                errorMsg = errorData.error || errorMsg;
            } catch (e) { /* ignore */ }
            throw new Error(errorMsg);
        }

        // Başarılı yanıtı işle
        const result = await response.json();
        resultSpan.textContent = ` - Tahmin: ${result.prediction}`;
        resultSpan.style.color = 'green';

    } catch (error) {
        console.error('Tahmin isteği sırasında hata:', error);
        resultSpan.textContent = ` - Hata: ${error.message}`;
        resultSpan.style.color = 'red';
    } finally {
        // Butonu tekrar etkinleştir
        buttonElement.disabled = false;
    }
}

// --- Sayfa Yüklendiğinde Çalışacak İlk Kodlar ---
document.addEventListener('DOMContentLoaded', () => {
    populateFormationSelect(); // Açılır listeyi doldur
    drawCanvas(); // Canvas'ı ilk kez çiz
    updateInfoArea("Başlamak için canvas'a tıklayın."); // Başlangıç mesajı
});