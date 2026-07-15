import pandas as pd
import os

def run_preprocessing():
    print("Memulai proses otomatisasi preprocessing...")
    
    # 1. Pastikan folder output tersedia
    output_dir = 'namadataset_preprocessing'
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Load raw data
    raw_path = 'namadataset_raw/Produksi Tanaman.csv'
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"File {raw_path} tidak ditemukan!")
        
    df = pd.read_csv(raw_path)
    
    # 3. Data Cleaning & Preprocessing
    # Mengganti karakter '-' menjadi angka 0 (karena BPS sering pakai '-' untuk nol)
    df = df.replace('-', 0)
    
    # Memisahkan kolom kategorikal (Provinsi) dan numerikal (Produksi)
    kolom_kategori = ['Provinsi']
    kolom_numerik = df.columns.drop(kolom_kategori)
    
    # Mengubah semua kolom produksi menjadi numerik
    df[kolom_numerik] = df[kolom_numerik].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # (Opsional) Lakukan Label Encoding untuk Provinsi jika model Anda membutuhkannya
    # df['Provinsi_Encoded'] = df['Provinsi'].astype('category').cat.codes
    
    # 4. Simpan data yang sudah bersih
    output_path = os.path.join(output_dir, 'data_clean.csv')
    df.to_csv(output_path, index=False)
    print(f"Preprocessing selesai! Data siap latih disimpan di: {output_path}")

if __name__ == "__main__":
    run_preprocessing()
