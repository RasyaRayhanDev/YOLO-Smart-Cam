# ☕ Cafe Analytics Dashboard

Sistem tracking pengunjung cafe dengan AI yang menampilkan analytics lengkap: rata-rata pengunjung per hari, rata-rata waktu yang dihabiskan, dan visualisasi data.

## Fitur
- 🤖 Deteksi orang menggunakan YOLO26 (model terbaru & tercepat)
- 🏃 **Activity Classification** - Deteksi aktivitas: Duduk, Berdiri, Berjalan
- 🎯 **Pose Estimation** - Tracking keypoints tubuh untuk analisis aktivitas
- 🔄 Re-identification (ReID) - mengenali orang yang sama meski keluar-masuk frame
- 📊 Dashboard analytics dengan grafik interaktif
- ⏱️ Tracking durasi waktu setiap pengunjung
- 💾 Data tersimpan otomatis untuk analisis jangka panjang
- 📹 Support webcam dan video file

## Instalasi

```bash
pip install -r requirements.txt
```

## Cara Menggunakan

```bash
streamlit run cafe_analytics.py
```

Browser akan otomatis terbuka di `http://localhost:8501`

## Dashboard

### Tab Analytics 📊
- **Rata-rata Pengunjung/Hari** - Berapa orang datang per hari (7 hari terakhir)
- **Rata-rata Waktu di Cafe** - Berapa lama pengunjung stay (dalam menit)
- **Total Pengunjung** - Total keseluruhan pengunjung
- **Grafik Pengunjung Harian** - Visualisasi tren pengunjung
- **Distribusi Durasi** - Histogram berapa lama orang biasanya stay
- **Tabel Detail** - Data lengkap per tanggal

### Tab Live Tracking 📹
- Live tracking dengan ReID dan Activity Classification
- Metrics real-time: 
  - Pengunjung saat ini
  - Total pengunjung hari ini
  - **Aktivitas breakdown** (Sitting/Standing/Walking)
- Color-coded bounding boxes:
  - 🟠 Orange = Sitting
  - 🟢 Green = Standing  
  - 🔵 Blue = Walking
  - ⚪ Gray = Unknown
- Data otomatis tersimpan ke `cafe_data.json`

## Cara Kerja ReID

**Re-identification** mengenali orang yang sama meskipun mereka keluar dan masuk kembali ke frame kamera.

1. Orang pertama kali terdeteksi → sistem simpan wajahnya
2. Orang keluar dari frame → ID tracking hilang tapi data wajah tersimpan
3. Orang masuk lagi → sistem cocokkan wajah dengan database
4. Jika match → pakai Person ID yang sama, waktu terus jalan
5. Jika tidak match → buat Person ID baru

**Perfect untuk cafe:** Pelanggan bisa pindah-pindah tempat atau keluar sebentar, tapi waktu tracking tetap akurat!

## Data Storage

### Person Embeddings Database
- **File**: `person_embeddings.pkl`
- Menyimpan face embeddings dan color histograms
- Persistent across sessions - tidak hilang saat restart
- Multiple embeddings per person (max 5 angles/poses)
- Auto-save saat person baru terdeteksi

### Analytics Data
- **File**: `cafe_data.json`
- Menyimpan data pengunjung harian dan durasi
- Format JSON untuk export mudah

### Reset Database
Jika ingin mulai dari awal (hapus semua data):
```bash
python reset_database.py
```

## Cara Kerja ReID (Improved)

**1. Persistent Database** 🗄️
- Embeddings tersimpan di file
- Saat restart, sistem ingat orang yang pernah datang
- Tidak perlu re-learn setiap kali

**2. Multiple Embeddings per Person** 📸
- Sistem simpan 5 angle/pose berbeda per orang
- Matching lebih akurat karena punya banyak referensi
- Auto-update saat orang terdeteksi dari angle baru

**3. Adaptive Threshold** 🎯
- Threshold turun otomatis kalau punya banyak embeddings
- Makin banyak data = makin mudah match
- Balance antara akurasi dan fleksibilitas

## Troubleshooting

### Error: Tidak bisa membuka webcam
- Pastikan webcam terhubung
- Coba restart aplikasi
- Cek permission webcam di browser/OS

### Model download otomatis
- Saat pertama kali run, YOLO26 akan download 2 model:
  - `yolo26n.pt` (~6MB) - Person detection
  - `yolo26n-pose.pt` (~7MB) - Pose estimation
- Tunggu hingga download selesai
- YOLO26 15% lebih cepat dari YOLO11 dengan akurasi lebih tinggi

### ReID lambat
- Normal, karena proses face recognition
- Gunakan video dengan resolusi lebih kecil jika terlalu lambat
- Pastikan wajah terlihat jelas untuk matching yang akurat

## Kustomisasi

### Mengubah Similarity Threshold
Edit di `cafe_analytics.py`:
```python
tracker = CafeTrackerStreamlit(similarity_threshold=0.7)  # Lebih strict
tracker = CafeTrackerStreamlit(similarity_threshold=0.5)  # Lebih loose
```

### Mengubah Timeout
Edit timeout untuk menganggap pengunjung sudah selesai (default 10 detik):
```python
if current_time - data["last_seen"] > 10:  # Ubah 10 ke nilai lain
```

## Activity Classification 🏃

Sistem menggunakan YOLO Pose untuk mendeteksi keypoints tubuh dan mengklasifikasikan aktivitas:

**Cara Kerja:**
1. YOLO Pose detect 17 keypoints (nose, shoulders, hips, knees, ankles, dll)
2. Analisis posisi relatif keypoints
3. Klasifikasi berdasarkan rasio panjang kaki vs torso:
   - **Sitting**: Leg bend ratio < 0.5 (kaki tertekuk)
   - **Standing**: Leg bend ratio 0.5-1.2 (kaki lurus, posisi tegak)
   - **Walking**: Leg bend ratio > 1.2 (kaki bergerak)

**Visualisasi:**
- Bounding box berwarna sesuai aktivitas
- Label menampilkan: Person ID | Activity | Duration

## Tech Stack

- **YOLO26** - Object detection (person detection)
- **YOLO26-Pose** - Pose estimation & activity classification
- **DeepFace** - Face recognition untuk ReID
- **Streamlit** - Web dashboard
- **Plotly** - Interactive charts
- **OpenCV** - Video processing

## Referensi
- [Ultralytics YOLO26](https://docs.ultralytics.com/)
- [DeepFace](https://github.com/serengil/deepface)
- [Streamlit](https://streamlit.io/)
