from google.colab import drive

drive.mount('/content/drive')

import nltk
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import re
import nltk
from nltk import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

val_data = pd.read_csv("/content/drive/MyDrive/Twitter Sentiment Analysis/twitter/twitter_validation.csv")
train_data = pd.read_csv("/content/drive/MyDrive/Twitter Sentiment Analysis/twitter/twitter_training.csv")
names = ['id', 'information', 'Sentiment', 'Tweet_content']

val_data.shape
train_data.shape
val_data.head()
train_data.head()

train_data.columns = ['id', 'information', 'Sentiment', 'Tweet_content']
train_data.head()

val_data.columns = ['id', 'information', 'Sentiment', 'Tweet_content']
val_data.head()

train_data["lower"] = train_data.Tweet_content.str.lower()  # lowercase
train_data["lower"] = [str(data) for data in train_data.lower]  # converting all to string
train_data["lower"] = train_data.lower.apply(lambda x: re.sub('[^A-Za-z0-9 ]+', ' ', x))  # regex
val_data["lower"] = val_data.Tweet_content.str.lower()  # lowercase
val_data["lower"] = [str(data) for data in val_data.lower]  # converting all to string
val_data["lower"] = val_data.lower.apply(lambda x: re.sub('[^A-Za-z0-9 ]+', ' ', x))  # regex

train_data.head()

train_data.duplicated().sum()
train_data.drop_duplicates(inplace=True)
train_data.duplicated().sum()
train_data.isnull().sum()
train_data.dropna(axis=0, inplace=True)
train_data.isnull().sum()
train_data.reset_index(inplace=True)
train_data.shape
val_data.duplicated().sum()
val_data.drop_duplicates(inplace=True)
val_data.duplicated().sum()
val_data.isnull().sum()
val_data.dropna(axis=0, inplace=True)
val_data.isnull().sum()
val_data.reset_index(inplace=True)
val_data.shape

word_cloud_text = ''.join(train_data[train_data["Sentiment"] == "Positive"].lower)
# Creation of wordcloud
wordcloud = WordCloud(
    max_font_size=100,
    max_words=100,
    background_color="black",
    scale=10,
    width=800,
    height=800
).generate(word_cloud_text)
# Figure properties
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

word_cloud_text = ''.join(train_data[train_data["Sentiment"] == "Negative"].lower)
# Creation of wordcloud
wordcloud = WordCloud(
    max_font_size=100,
    max_words=100,
    background_color="black",
    scale=10,
    width=800,
    height=800
).generate(word_cloud_text)
# Figure properties
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

word_cloud_text = ''.join(train_data[train_data["Sentiment"] == "Irrelevant"].lower)
# Creation of wordcloud
wordcloud = WordCloud(
    max_font_size=100,
    max_words=100,
    background_color="black",
    scale=10,
    width=800,
    height=800
).generate(word_cloud_text)
# Figure properties
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

word_cloud_text = ''.join(train_data[train_data["Sentiment"] == "Neutral"].lower)
# Creation of wordcloud
wordcloud = WordCloud(
    max_font_size=100,
    max_words=100,
    background_color="black",
    scale=10,
    width=800,
    height=800
).generate(word_cloud_text)
# Figure properties
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

plot1 = train_data.groupby(by=["information", "Sentiment"]).count().reset_index()
plot1.head()

plt.figure(figsize=(20, 6))
sns.barplot(data=plot1, x="information", y="id", hue="Sentiment")
plt.xticks(rotation=90)
plt.xlabel("Brand")
plt.ylabel("Number of tweets")
plt.grid()
plt.title("Distribution of tweets per Branch and Sentiment");

tokens_text = [word_tokenize(str(word)) for word in train_data.lower]

tokens_counter = [item for sublist in tokens_text for item in sublist]
print("Number of tokens: ", len(set(tokens_counter)))

tokens_text[1]

stopwords_nltk = nltk.corpus.stopwords
stop_words = stopwords_nltk.words('english')
stop_words[:5]


def clean_text(text):
    text = text.lower()
    for s in tokens_text:
        text = text.replace(s, tokens_text[s])
    text = ' '.join(text.split())
    return text


x = train_data['Tweet_content']
x[20]

bow_counts = CountVectorizer(
    tokenizer=word_tokenize,
    stop_words=stop_words,
    ngram_range=(1, 1)
)

reviews_train, reviews_test = train_test_split(train_data, test_size=0.2, random_state=0)
X_train_bow = bow_counts.fit_transform(reviews_train.lower)
X_test_bow = bow_counts.transform(reviews_test.lower)
X_test_bow

y_train_bow = reviews_train['Sentiment']
y_test_bow = reviews_test['Sentiment']

y_test_bow.value_counts() / y_test_bow.shape[0]

LR = LogisticRegression(solver="liblinear")
LR.fit(X_train_bow, y_train_bow)

test_pred = LR.predict(X_test_bow)
print("Accuracy: ", accuracy_score(y_test_bow, test_pred))

X_val_bow = bow_counts.transform(val_data.lower)
y_val_bow = val_data['Sentiment']

X_val_bow
Val_res = LR.predict(X_val_bow)
print("Accuracy: ", accuracy_score(y_val_bow, Val_res))

bow_counts = CountVectorizer(
    tokenizer=word_tokenize,
    ngram_range=(1, 4)
)

X_train_bow = bow_counts.fit_transform(reviews_train.lower)
X_test_bow = bow_counts.transform(reviews_test.lower)
X_val_bow = bow_counts.transform(val_data.lower)

X_train_bow
LR2 = LogisticRegression(solver="liblinear")

LR2.fit(X_train_bow, y_train_bow)

test_pred_2 = LR2.predict(X_test_bow)
print("Accuracy: ", accuracy_score(y_test_bow, test_pred_2))

y_val_bow = val_data['Sentiment']
Val_pred_2 = LR2.predict(X_val_bow)
print("Accuracy: ", accuracy_score(y_val_bow, Val_pred_2))

from nltk.stem.porter import PorterStemmer
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, GlobalMaxPooling1D, SpatialDropout1D

import re

REPLACE_WITH_SPACE = re.compile("(@)")
SPACE = " "
from nltk.corpus import stopwords

english_stop_words = stopwords.words('english')
from nltk.stem.porter import PorterStemmer


# 1
def reviews(reviews):
    reviews = [REPLACE_WITH_SPACE.sub(SPACE, line.lower()) for line in reviews]

    return reviews


# 2
def remove_stop_words(corpus):
    removed_stop_words = []
    for review in corpus:
        removed_stop_words.append(
            ' '.join([word for word in review.split() if word not in english_stop_words]))
    return removed_stop_words


# 3
def get_stemmed_text(corpus):
    stemmer = PorterStemmer()

    return [' '.join([stemmer.stem(word) for word in review.split()]) for review in corpus]


y = train_data['Sentiment']

# 1
reviewtweet = reviews(x)
# 2
no_stop_words_tweet = remove_stop_words(reviewtweet)
# 3
stemmed_reviews_tweet = get_stemmed_text(no_stop_words_tweet)

stemmed_reviews_tweet[20]

max_words = 8000

tokenizer = Tokenizer(
    num_words=max_words,
    filters='"#$%&()*+-/:;<=>@[\]^_`{|}~'
)
tokenizer.fit_on_texts(stemmed_reviews_tweet)
x = tokenizer.texts_to_sequences(stemmed_reviews_tweet)
x = pad_sequences(x, maxlen=300)
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(y)

y = np.array(label_tokenizer.texts_to_sequences(y))

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

model_lstm = Sequential()
model_lstm.add(Embedding(input_dim=max_words, output_dim=128, input_length=300))
model_lstm.add(SpatialDropout1D(0.3))
model_lstm.add(LSTM(128, dropout=0.3, recurrent_dropout=0.3))
model_lstm.add(Dense(128, activation='relu'))
model_lstm.add(Dropout(0.3))
model_lstm.add(Dense(5, activation='softmax'))
model_lstm.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='Adam',
    metrics=['accuracy']
)

history = model_lstm.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    epochs=10,
    batch_size=512
)

model_lstm.summary()

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()