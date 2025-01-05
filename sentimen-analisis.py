import streamlit as st
import pickle
import pandas as pd
import re
import emoji
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Load the saved model and vectorizer
filename = 'svm_model.sav'
filename_vectorizer = 'tfidf_vectorizer.sav'
loaded_model = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open(filename_vectorizer, 'rb'))

def remove_mention(text):
    text = re.sub(r'@\w+', '', text)
    return text

def remove_retweet(text):
    text = re.sub(r'RT\s', '', text)
    return text

def remove_urls(text):
    return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def remove(text):
    text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())
    return text

# Kamus emotikon ke kata-kata dalam bahasa Indonesia
emoticon_dict = {
    ":)": "senyum",
    ":-)": "senyum",
    ":(": "sedih",
    ":-(": "sedih",
    ":D": "senyum lebar",
    ":-D": "senyum lebar",
    ":P": "muka usil",
    ";)": "kedip",
    "<3": "hati",
    ":o": "terkejut",
    ":O": "terkejut"
}

def convert_emoticons_to_words(text):
    # Ubah setiap emotikon ke kata-kata menggunakan kamus
    for emoticon, word in emoticon_dict.items():
        text = re.sub(re.escape(emoticon), word, text)
    return text

def convert_emoji_to_words(text):
    # Mengonversi emoji menjadi teks deskriptif
    return emoji.demojize(text, language="id")  # "id" untuk bahasa Indonesia

def remove_punct(text):
    text = "".join([char for char in text if char not in string.punctuation])
    return text

def to_lowercase(text):
    # Mengonversi teks menjadi huruf kecil
    return text.lower()

# Kamus normalisasi untuk kata-kata gaul atau typo
normalization_dict = {
    "gimana": "bagaimana",
    "gpp": "tidak apa-apa",
    "gapapa": "tidak apa-apa",
    "gk": "tidak",
    "gak": "tidak",
    "nggak": "tidak",
    "tdk": "tidak",
    "bgt": "banget",
    "bgt": "sekali",
    "sy": "saya",
    "aq": "aku",
    "udh": "sudah",
    "sdh": "sudah",
    "sdh": "sudah",
    "aja": "saja",
    "aj": "saja",
    "sm": "sama",
    "sama2": "sama-sama",
    "kpn": "kapan",
    "kalo": "kalau",
    "klo": "kalau",
    "trs": "terus",
    "trus": "terus",
    "tp": "tapi",
    "tpi": "tapi",
    "tapi": "tetapi",
    "dgn": "dengan",
    "dg": "dengan",
    "dr": "dari",
    "dri": "dari",
    "utk": "untuk",
    "untk": "untuk",
    "ya": "iya",
    "iyaa": "iya",
    "iye": "iya",
    "jg": "juga",
    "jga": "juga",
    "jd": "jadi",
    "jdi": "jadi",
    "mn": "mana",
    "mna": "mana",
    "km": "kamu",
    "mw": "mau",
    "mau": "ingin",
    "yng": "yang",
    "bs": "bisa",
    "bisa": "dapat",
    "udh": "sudah",
    "blm": "belum",
    "blm": "belum",
    "bnyk": "banyak",
    "bnyak": "banyak",
    "sdng": "sedang",
    "org": "orang",
    "krn": "karena",
    "karna": "karena",
    "kek": "seperti",
    "kyk": "seperti",
    "kayak": "seperti",
    "gtu": "begitu",
    "gini": "begini",
    "lu": "kamu",
    "loe": "kamu",
    "gua": "aku",
    "gue": "aku",
    "apaan": "apa",
    "apaansih": "apa sih",
    "knapa": "kenapa",
    "ngapain": "mengapa",
    "dpt" : "dapat",
    "sih" : "",
    "yt" : "youtube",
    "yuk": "ayo",
    "yuks": "ayo",
    "sssaja": "saja",
    "rp" : "rupiah"
}

def normalize_text(text):
    words = text.split()
    normalized_words = [normalization_dict.get(word, word) for word in words]
    return " ".join(normalized_words)

stop_words = set(stopwords.words('indonesian')) 
def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)

def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens

factory = StemmerFactory()
stemmer = factory.create_stemmer()
def stem_text(tokens):
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens

def text_preprocessing_process(text):
    text = remove_mention(text)
    text = remove_retweet(text)
    text = remove_urls(text)
    text = remove_html_tags(text)
    text = remove(text)
    text = convert_emoticons_to_words(text)
    text = convert_emoji_to_words(text)
    text = remove_punct(text)
    text = to_lowercase(text)
    text = normalize_text(text)
    text = remove_stopwords(text)
    text = tokenize_text(text)
    text = stem_text(text)
    return ' '.join(text)

def determine_sentiment(text):
    """Perform sentiment analysis on the given text using the loaded model."""
    # Preprocess the input text
    processed_text = text_preprocessing_process(text)
    # Vectorize the processed text
    text_vectorized = vectorizer.transform([processed_text])

    # Predict using the loaded model
    prediction = loaded_model.predict(text_vectorized)[0]
    return prediction


# Streamlit app
st.title("Sentiment Analysis App")

# Input text area
tweet_input = st.text_area("Enter your tweet:", "Type your tweet here...")

# Perform sentiment analysis when a button is pressed
if st.button("Analyze"):
  sentiment = determine_sentiment(tweet_input)
  st.write(f"Sentiment: {sentiment}")
