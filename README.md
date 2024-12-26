# Sentiment Analysis

This project performs sentiment analysis on Indonesian tweets using a pre-trained SVM model and lexicon-based approach.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/sentiment-analysis.git
    cd sentiment-analysis
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download NLTK stopwords:
    ```python
    import nltk
    nltk.download('stopwords')
    ```

## How to Run the Project

Run the Streamlit app using the following command:

```bash
streamlit run sentimen-analisis.py


### 2. **Menambahkannya dalam Kode**
Pastikan kode untuk mendownload stopwords ada di file Python:

```python
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


### 3. **Menggunakan Skrip Setup (opsional)**

Jika Anda ingin prosesnya otomatis saat pengguna menjalankan aplikasi, Anda bisa membuat skrip Python terpisah untuk setup:

Buat file `setup_nltk.py`:

```python
import nltk

nltk.download('stopwords')
nltk.download('punkt')

Run the following command to set up necessary NLTK resources:

```bash
python setup_nltk.py


### 4. **Periksa Ketersediaan Internet**

Pastikan server atau lingkungan tempat Anda menjalankan aplikasi memiliki akses internet, karena **nltk** memerlukan koneksi untuk mengunduh stopwords dari repositori online.

### 5. **Periksa Instalasi NLTK di Lingkungan Virtual**

Jika Anda menggunakan lingkungan virtual seperti `venv` atau `conda`, pastikan semua pustaka dan resource diunduh dalam lingkungan yang benar:

1. Aktifkan lingkungan virtual Anda.
2. Install dependencies (`pip install -r requirements.txt`).
3. Jalankan perintah `nltk.download('stopwords')` dalam lingkungan tersebut untuk memastikan resource terinstal di lokasi yang tepat.

Dengan langkah-langkah ini, seharusnya masalah **stopwords** dapat teratasi dan tidak ada error lagi.
