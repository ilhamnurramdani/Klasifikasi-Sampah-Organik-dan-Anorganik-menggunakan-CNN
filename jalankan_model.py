import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# --- 1. MEMUAT MODEL YANG SUDAH JADI ---
# Pastikan file 'model_sampah.h5' ada di folder yang sama dengan script ini
path_model = 'model_sampah.h5'

if not os.path.exists(path_model):
    print(f"❌ Error: File '{path_model}' tidak ditemukan!")
    print("Pastikan Anda sudah menjalankan training setidaknya sekali.")
    exit()

print("Sedang memuat model... Mohon tunggu.")
model = tf.keras.models.load_model(path_model)
print("✅ Model berhasil dimuat!")

# --- 2. FUNGSI PREDIKSI ---
def prediksi_gambar(path_gambar):
    try:
        # Load gambar dan ubah ukurannya menjadi 150x150 (sesuai saat training)
        img = image.load_img(path_gambar, target_size=(150, 150))
        
        # Ubah gambar menjadi array numpy
        x = image.img_to_array(img)
        
        # Tambahkan dimensi batch (karena model mengharapkan input bentuk (1, 150, 150, 3))
        x = np.expand_dims(x, axis=0)
        
        # PENTING: Normalisasi gambar (0-1) karena saat training kita pakai rescale=1./255
        x = x / 255.0

        # Lakukan prediksi
        classes = model.predict(x, verbose=0)
        
        # Ambil nilai probabilitas
        nilai_prediksi = classes[0][0]
        
        # LOGIKA KELAS (Berdasarkan urutan Abjad Folder: Inorganic, Organic)
        # 0 = Inorganic
        # 1 = Organic
        # Ambang batas (Threshold) biasanya 0.5
        
        print(f"\n--- HASIL PREDIKSI ---")
        print(f"File: {path_gambar}")
        print(f"Skor Probabilitas: {nilai_prediksi:.4f}")
        
        if nilai_prediksi > 0.5:
            print("Kategori: 🌿 ORGANIC (Sampah Organik)")
        else:
            print("Kategori: 🥤 INORGANIC (Sampah Anorganik)")
            
    except Exception as e:
        print(f"Error saat memproses gambar: {e}")

# --- 3. LOOP UTAMA (Agar bisa tes berkali-kali) ---
while True:
    print("\n" + "="*30)
    nama_file = input("Masukkan nama file gambar (atau ketik 'x' untuk keluar): ")
    
    if nama_file.lower() == 'x':
        print("Program selesai.")
        break
    
    # Cek apakah file ada
    if os.path.exists(nama_file):
        prediksi_gambar(nama_file)
    else:
        print("⚠️ File tidak ditemukan. Pastikan nama file benar dan ada ekstensi (misal .jpg)")