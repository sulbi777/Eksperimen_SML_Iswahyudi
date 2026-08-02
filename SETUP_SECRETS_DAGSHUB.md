# Setup DagsHub dan GitHub Actions

## DagsHub untuk Kriteria 2 advanced

1. Buat repository DagsHub publik, misalnya `USERNAME/bank-marketing-mlops`.
2. Di DagsHub buka **Repository > Remote > MLflow** lalu salin tracking URI dan ikuti login/token yang ditampilkan.
3. Buat akses token DagsHub. Jangan pernah commit token.
4. Untuk menjalankan lokal, isi environment variable Windows PowerShell (hanya sesi aktif):

```powershell
$env:DAGSHUB_USER="USERNAME"
$env:DAGSHUB_REPO="bank-marketing-mlops"
$env:MLFLOW_TRACKING_USERNAME="USERNAME"
$env:MLFLOW_TRACKING_PASSWORD="TOKEN_DAGSHUB"
python Membangun_model/modelling_tuning.py
```

Jika variabel DagsHub tidak diisi, skrip tetap berjalan dengan MLflow lokal pada folder `mlruns`. Setelah berhasil, salin URL experiment/run ke `Membangun_model/DagsHub.txt`.

## GitHub Secrets untuk training langsung dari GitHub

Di **Settings > Secrets and variables > Actions**, buat empat Repository secrets berikut agar workflow `train-and-track-dagshub` dapat mengirim metrik dan artefak ke DagsHub:

| Nama Secret | Nilai |
| --- | --- |
| `DAGSHUB_USER` | username DagsHub Anda |
| `DAGSHUB_REPO` | nama repository DagsHub, tanpa username |
| `MLFLOW_TRACKING_USERNAME` | username DagsHub Anda |
| `MLFLOW_TRACKING_PASSWORD` | token akses DagsHub |

Setelah itu buka **Actions > train-and-track-dagshub > Run workflow**. Mulailah dengan `n_iter=12`; GitHub Actions akan mengunduh data, melakukan preprocessing, tuning, lalu mencatat run ke DagsHub.

## GitHub Actions preprocessing

Workflow preprocessing tidak memerlukan token tambahan: `GITHUB_TOKEN` bawaan dipakai untuk commit dataset hasil preprocess. Pada repository GitHub, buka **Settings > Actions > General > Workflow permissions**, pilih **Read and write permissions**, lalu Save. Trigger lewat **Actions > preprocess-bank-marketing > Run workflow**.

## Rahasia yang dilarang masuk Git

Simpan `MLFLOW_TRACKING_PASSWORD`, Docker Hub token, dan token API apa pun sebagai Environment variable atau GitHub Secret. File `.env`, token, dan `mlruns` sudah diabaikan oleh `.gitignore`.
