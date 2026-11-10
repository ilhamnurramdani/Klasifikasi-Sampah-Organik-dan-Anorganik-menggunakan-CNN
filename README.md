# ♻️ Waste Classification Using Convolutional Neural Network (CNN)

<p align="center">
  <b>Image Classification for Organic and Inorganic Waste</b>
</p>

<p align="center">
  A Deep Learning project to classify waste images into Organic and Inorganic categories using Convolutional Neural Network (CNN) with TensorFlow.
</p>

---

## 📌 Project Overview

Proyek ini merupakan implementasi **Deep Learning** untuk mengklasifikasikan gambar sampah ke dalam dua kategori, yaitu **Organic (Sampah Organik)** dan **Inorganic (Sampah Anorganik)** menggunakan metode Convolutional Neural Network (CNN).

Model CNN dirancang untuk mempelajari karakteristik visual dari gambar, seperti bentuk, tekstur, dan pola, sehingga dapat mengenali serta membedakan kedua jenis sampah secara otomatis.

Proyek ini menggunakan TensorFlow dan Keras untuk membangun, melatih, serta menguji model klasifikasi gambar.

## 🎯 Project Objectives

- Mengembangkan model klasifikasi gambar menggunakan algoritma CNN.
- Mengidentifikasi sampah organik dan anorganik secara otomatis.
- Menerapkan image preprocessing dan data augmentation.
- Mengevaluasi performa model berdasarkan training dan validation.
- Mengimplementasikan model yang telah dilatih untuk melakukan prediksi gambar baru.

## ✨ Key Features

- **Image Classification:** Mengklasifikasikan gambar menjadi dua kategori sampah.
- **Convolutional Neural Network:** Menggunakan arsitektur CNN untuk ekstraksi fitur gambar.
- **Image Preprocessing:** Melakukan normalisasi gambar sebelum proses training.
- **Data Augmentation:** Menggunakan rotation, horizontal flip, dan shear untuk memperkaya variasi data training.
- **Model Training:** Melatih model menggunakan TensorFlow dan Keras.
- **Image Prediction:** Melakukan prediksi terhadap gambar sampah baru.
- **Training Visualization:** Menampilkan grafik accuracy dan loss selama proses training.

## 🛠️ Tech Stack

| Technology | Description |
|---|---|
| Python | Main programming language |
| TensorFlow | Deep Learning framework |
| Keras | Neural network implementation |
| NumPy | Numerical computation |
| Matplotlib | Data visualization |
| CNN | Image classification algorithm |

## 🧠 CNN Architecture

Model menggunakan arsitektur Convolutional Neural Network dengan konfigurasi berikut:

| Layer | Configuration |
|---|---|
| Input | 150 × 150 × 3 |
| Conv2D | 32 filters, 3 × 3, ReLU |
| MaxPooling2D | 2 × 2 |
| Conv2D | 64 filters, 3 × 3, ReLU |
| MaxPooling2D | 2 × 2 |
| Conv2D | 128 filters, 3 × 3, ReLU |
| MaxPooling2D | 2 × 2 |
| Flatten | Feature flattening |
| Dense | 512 neurons, ReLU |
| Dropout | 0.5 |
| Output | 1 neuron, Sigmoid |

**Model Configuration:**
- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Evaluation Metric: Accuracy
- Batch Size: 32
- Epochs: 10
- Image Size: 150 × 150 pixels

## 📂 Project Structure

```text
Klasifikasi-Sampah-CNN/
│
├── dataset/
│   ├── train/
│   │   ├── Inorganic/
│   │   └── Organic/
│   │
│   └── test/
│       ├── Inorganic/
│       └── Organic/
│
├── train.py
├── predict.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── model_klasifikasi_sampah.h5
```

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/ilhamnurramdani/Klasifikasi-Sampah-Organik-dan-Anorganik-menggunakan-CNN.git
```

### 2. Navigate to Project Directory

```bash
cd Klasifikasi-Sampah-Organik-dan-Anorganik-menggunakan-CNN
```

### 3. Create Virtual Environment

```bash
python -m venv .env
```

Aktifkan virtual environment pada Windows:

```bash
.env\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## 🚀 How to Run

### 1. Prepare Dataset

Pastikan dataset memiliki struktur folder seperti berikut:

```text
dataset/
├── train/
│   ├── Inorganic/
│   └── Organic/
│
└── test/
    ├── Inorganic/
    └── Organic/
```

### 2. Train the Model

Jalankan perintah:

```bash
python train.py
```

Proses training meliputi:
- Image preprocessing dan normalization.
- Data augmentation.
- Feature extraction menggunakan CNN.
- Model training selama 10 epochs.
- Visualization training accuracy dan loss.
- Model saving.

Model hasil training disimpan sebagai:

```text
model_klasifikasi_sampah.h5
```

### 3. Run Image Prediction

Pastikan model sudah berhasil dilatih, kemudian jalankan:

```bash
python predict.py
```

Masukkan nama atau path gambar yang ingin diklasifikasikan.

Contoh:

```text
Masukkan nama file gambar: sampah_plastik.jpg

--- HASIL PREDIKSI ---
Kategori: INORGANIC (Sampah Anorganik)
```

Ketik `x` untuk keluar dari program.

## 📈 Training Visualization

Selama proses training, sistem menghasilkan dua grafik utama:

**1. Training and Validation Accuracy**

Digunakan untuk melihat perkembangan akurasi model pada data training dan validation.

**2. Training and Validation Loss**

Digunakan untuk mengamati penurunan nilai loss serta melihat indikasi overfitting selama proses pembelajaran.

## 📊 Model Evaluation

Evaluasi model dilakukan dengan mengamati performa training dan validation, khususnya melalui nilai accuracy dan loss pada setiap epoch.

Hasil evaluasi akhir dapat ditambahkan setelah eksperimen training dilakukan.

## 🔮 Future Improvements

Beberapa pengembangan yang dapat dilakukan pada proyek ini:

- Menambahkan kategori sampah yang lebih beragam.
- Meningkatkan performa model melalui hyperparameter tuning.
- Membandingkan CNN dengan arsitektur transfer learning seperti MobileNetV2 dan VGG16.
- Mengembangkan aplikasi berbasis web untuk klasifikasi gambar secara interaktif.
- Mengimplementasikan confusion matrix, precision, recall, dan F1-score.

## 👨‍💻 Author

**Ilham Nur Ramdani**

Master of Computer Science Student  
Universitas Budi Luhur

**Areas of Interest:**
- Artificial Intelligence
- Machine Learning
- Deep Learning
- Data Science
- Data Mining
- Web Development

GitHub: [@ilhamnurramdani](https://github.com/ilhamnurramdani)

---

<p align="center">
  <b>Built with Python & TensorFlow</b>
  <br>
  ⭐ If you find this project useful, feel free to give it a star!
</p>
