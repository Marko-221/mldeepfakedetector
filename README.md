# ML_DeepFakeDetector
# Aplikasi Detektor Deepfake (Gambar & Video)

> Proyek ini adalah aplikasi web yang mampu mendeteksi apakah sebuah gambar atau video wajah merupakan hasil manipulasi **Deepfake** atau **Asli**. Aplikasi ini menggunakan model *deep learning* yang telah dilatih (MobileNetV2) untuk menganalisis file yang diunggah dan memberikan probabilitas keasliannya.
---

## 👥 Anggota Kelompok
* Marco Darian Thomas(221112216) => Train Model
* Hugo Edri Chandra(221111848) => FrontEnd & Backend
* Valentino Karada(221110851) => Dokumentasi
---

## 🛠️ Teknologi yang Digunakan
Proyek ini dibangun menggunakan tumpukan teknologi berikut:
* **Backend:**
    * **Python 3.9+**
    * **Flask:** Sebagai *micro-framework* web untuk melayani API dan frontend.
    * **TensorFlow / Keras:** Untuk memuat dan menjalankan model *deep learning*.
    * **OpenCV (cv2):** Untuk membaca, membongkar, dan memproses file video *frame-by-frame*.
    * **Pillow (PIL):** Untuk memproses file gambar.
    * **Numpy:** Untuk manipulasi data numerik dan *batch processing*.
* **Model AI:**
    * **MobileNetV2** (via Transfer Learning) yang telah di-*fine-tuning* pada dataset deteksi deepfake.
* **Frontend:**
    * HTML5
    * CSS3
    * JavaScript (fetch API)
---

## 🚀 Petunjuk Penggunaan Aplikasi
Aplikasi ini sangat mudah digunakan dan dapat mendeteksi gambar maupun video:
1.  Buka aplikasi di browser (secara lokal di `http://127.0.0.1:5000`).
2.  Klik tombol **"Pilih File (Gambar/Video)"**.
3.  Pilih file `.jpg`, `.png`, atau `.mp4` dari komputer Anda.
4.  Pratinjau gambar atau video akan muncul di layar.
5.  Klik tombol **"Deteksi!"** untuk memulai analisis.
6.  Harap tunggu:
    * **Jika gambar:** Hasil akan muncul dalam beberapa detik.
    * **Jika video:** Proses akan memakan waktu lebih lama (bisa 10-30 detik) karena backend perlu menganalisis 30 frame dari video tersebut.
7.  Hasil akhir (**Palsu (Fake)** atau **Asli (Real)**) akan ditampilkan beserta tingkat keyakinannya.
---

## ⚙️ Instalasi & Menjalankan Proyek di Lokal
Ikuti langkah-langkah ini untuk menginstal dan menjalankan salinan proyek ini di komputer lokal Anda.

### Prasyarat
Pastikan perangkat Anda telah terinstal perangkat lunak berikut:
* Git
* Python 3.9 atau yang lebih baru
* `pip` (Manajer paket Python)

### 1. Instalasi
1.  **Clone Repositori**
    Buka terminal Anda dan clone repositori ini:
    ```bash
    git clone https://github.com/KaradaVal/ML_DeepFakeDetector.git
    ```
2.  **Masuk ke Direktori Proyek**
    ```bash
    cd ML_DeepFakeDetector
    ```

3.  **Buat Virtual Environment (Sangat Direkomendasikan)**
    Ini akan mengisolasi dependensi proyek Anda.
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

4.  **Instal Dependensi**
    Pastikan Anda memiliki file `requirements.txt` di folder proyek Anda dan jalankan:
    ```bash
    pip install -r requirements.txt
    ```
    *(Jika Anda tidak memiliki `requirements.txt`, instal secara manual: `pip install flask tensorflow numpy pillow opencv-python-headless`)*

5.  **File Model**
    Pastikan file model Anda (`deepfake_detector_finetuned.h5`) ada di dalam folder utama proyek, di samping `app.py`.

### 2. Menjalankan Proyek

1.  **Jalankan Server Flask**
    Setelah semua dependensi terinstal dan `venv` aktif, jalankan perintah berikut di terminal:
    ```bash
    python app.py
    ```

2.  **Buka Aplikasi**
    Server akan berjalan dan Anda akan melihat output seperti ini:
    ```
     * Running on [http://127.0.0.1:5000](http://127.0.0.1:5000)
    ```
    Buka alamat `http://127.0.0.1:5000` di browser Anda untuk menggunakan aplikasi.
