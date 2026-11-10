import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 1. KONFIGURASI GPU (Wajib untuk RTX 3050) ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Terdeteksi: {gpus[0].name}")
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ GPU tidak terdeteksi (Running on CPU)")

# --- 2. PERSIAPAN DATASET ---
# Pastikan folder 'dataset' ada satu level dengan file python ini
base_dir = 'dataset' 
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'test') # Sesuai struktur folder Anda

# Cek apakah folder ada
if not os.path.exists(train_dir):
    print(f"❌ Error: Folder {train_dir} tidak ditemukan.")
    exit()

# Augmentasi Data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    shear_range=0.2,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Load Data
print("Memuat data...")
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# --- 3. MEMBANGUN MODEL (Sama persis dengan sebelumnya) ---
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# --- 4. PELATIHAN (TRAINING) ---
# Menggunakan dynamic steps agar aman berapapun jumlah data Anda
steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = validation_generator.samples // validation_generator.batch_size

# Jika data sangat sedikit (kurang dari batch size), set minimal 1
if steps_per_epoch == 0: steps_per_epoch = 1
if validation_steps == 0: validation_steps = 1

print("\nMulai Training...")
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=20, # Sesuaikan jumlah epoch (coba 20 dulu)
    validation_data=validation_generator,
    validation_steps=validation_steps,
    verbose=1 # Tampilkan progress bar
)

# Simpan model agar tidak perlu training ulang
model.save('model_sampah.h5')
print("✅ Model disimpan sebagai 'model_sampah.h5'")

# --- 5. VISUALISASI ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# --- 6. PREDIKSI GAMBAR BARU (PENGGANTI FILES.UPLOAD) ---
print("\n--- UJI COBA PREDIKSI ---")

# Ganti nama file ini dengan gambar yang ingin Anda tes
# Pastikan gambar tersebut ada di folder yang sama dengan script ini, atau tulis path lengkapnya
nama_file_gambar = "tes_botol.jpg" 

if os.path.exists(nama_file_gambar):
    path = nama_file_gambar
    img = image.load_img(path, target_size=(150, 150))
    
    # Tampilkan gambar yang diprediksi
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0 # Jangan lupa normalisasi manual saat prediksi!

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    
    # Cek label kelas dari generator (untuk memastikan urutan 0 dan 1)
    # Biasanya: {'inorganic': 0, 'organic': 1} atau sebaliknya (sesuai abjad folder)
    print(f"Class Indices: {train_generator.class_indices}")
    print(f"Nilai Prediksi: {classes[0][0]}")

    if classes[0][0] > 0.5:
        # Perlu dicek folder mana yang dianggap 1. Biasanya urut abjad.
        # Jika folder inorganic dan organic, I (Inorganic) = 0, O (Organic) = 1.
        # Tapi code asli Anda logika: > 0.5 adalah ANORGANIK. 
        # Mari kita ikuti logika output standar keras (Abjad):
        # 0: inorganic, 1: organic
        print(f"Hasil: {nama_file_gambar} adalah KELAS 1")
    else:
        print(f"Hasil: {nama_file_gambar} adalah KELAS 0")
else:
    print(f"⚠️ File '{nama_file_gambar}' tidak ditemukan. Silakan ganti variabel 'nama_file_gambar' di kode bawah.")