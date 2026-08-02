# Eksperimen_SML_Iswahyudi

Repository Kriteria 1 dan Kriteria 2 (advanced) untuk klasifikasi **UCI Bank Marketing**. Targetnya adalah prediksi apakah nasabah berlangganan deposito berjangka. Dataset berasal dari [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing).

## Struktur

- `Eksperimen_Iswahyudi.ipynb`: eksperimen manual - wajib dieksekusi dari atas ke bawah.
- `download_raw_data.py`: mengambil dataset mentah resmi UCI.
- `preprocessing/automate_Iswahyudi.py`: pipeline repeatable yang menghasilkan data siap latih.
- `.github/workflows/preprocess.yml`: otomatisasi preprocessing ketika dipicu.
- `Membangun_model/`: baseline MLflow dan tuning advanced + logging DagsHub.

## Mulai cepat

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python download_raw_data.py
python preprocessing/automate_Iswahyudi.py
python Membangun_model/modelling.py
python Membangun_model/modelling_tuning.py
```

Jalankan notebook setelah `download_raw_data.py`; notebook mengulang preprocessing secara manual sebagai bukti eksperimen. Untuk menghindari target leakage pada skenario sebelum kampanye, fitur `duration` tidak digunakan.

## Push pertama

```bash
git init -b main
git add .
git commit -m "feat: bank marketing experiment and MLOps"
git remote add origin https://github.com/USERNAME/Eksperimen_SML_Iswahyudi.git
git push -u origin main
```

Buat repository **Public** di GitHub tanpa README/license terlebih dahulu. Ikuti [SETUP_SECRETS_DAGSHUB.md](SETUP_SECRETS_DAGSHUB.md) sebelum mengaktifkan tracking online.
