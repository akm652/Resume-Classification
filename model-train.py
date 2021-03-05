# Import the dependencies
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import re

from transformers import pipeline
from nltk import *
from nltk.corpus import stopwords
from keras.layers import Dense, Dropout
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support

# fix random seed for reproducibility
np.random.seed(2)

# Loading the dataset
# pandas read_csv function to read csv dataset
df = pd.read_csv('./Data Collection + Data/Datasets/LinkedIn_ProfileDataset_Text_Role_200.csv')

# get info from pandas about this csv dataset
df.info()

# load nltk stopwords
stop = stopwords.words('english')


# Text preprocessing
# function to clean text from any punctuation, symbols and weird characters
def clean_up(text):
    text = text.replace(' ', ' ')
    text = text.replace('–', '-')
    text = text.replace('\n', '')
    text = re.sub(r'[IVXLCDM]+\.', '', text)
    text = re.sub(r'\[[^]]*\]', '', text)
    text = re.sub(r'\((.*?)\)', '', text)
    text = re.sub(r'\{(.*?)\}', '', text)
    text = re.sub(r'\b[A-Z]{2,}\b', '', text)
    text = text.replace(' ,', ',')
    text = text.replace(' .', '.')
    text = text.replace(' ;', ';')
    text = re.sub(r'[^A-Za-z0-9äÄöÖüÜß\s\.\-\!\?\:\;\,]', '', text)
    text = re.sub(r' +', ' ', text)
    text = text.replace('ä', 'ae')
    text = text.replace('ö', 'oe')
    text = text.replace('ü', 'ue')
    text = text.replace('Ä', 'Ae')
    text = text.replace('Ö', 'Oe')
    text = text.replace('Ü', 'Ue')
    text = text.replace('ß', 'ss')
    text = text.replace('See endorsements', '')
    text = re.sub(r' +', ' ', text)
    text = re.sub('[0-9]+', '', text)
    return text


# a regex syntax for removing all nltk stopwords in the text
stopword_regex = r'\b(?:{})\b'.format('|'.join(stop))

# using df.apply to apply regex syntax to dataframe column
df['Input'] = df['Input'].str.replace(stopword_regex, '')

# using df.apply to apply function to dataframe column
df['Role'] = df['Role'].apply(clean_up, 1)
df['Input'] = df['Input'].apply(clean_up, 1)

# using df.value_counts() from pandas to calculate amount of rows of data to every value of 'Role'
df.Role.value_counts()

# convert dataframe to numpy array
dataset = df.to_numpy()

# The dataset contains first column as labels (X) and second column as text input (Y)
# split into output (X) and input (y) variables
X = dataset[:, :-1]
Y = dataset[:, -1]


# function for splitting long sentences (huggingface summarization api doesn't take long input of text)
def split_sentences(text):
    words = text.split()

    length = len(words)

    print('Text original length is ' + str(length) + ' words')

    mid_index = length // 2

    if mid_index > 512:

        print('The original middle index of text is ' + str(mid_index))

        while mid_index > 512:
            print('Shortening text')
            mid_index = mid_index // 2
            print('The middle index of text is now ' + str(mid_index))
            length = mid_index * 2
            print('The length of text is now ' + str(length))

            if mid_index in range(512, 1024):
                print('The middle index exceeds range')
                length = mid_index
                print('The length of text is now ' + str(length))
                mid_index = mid_index // 2
                print('The middle index of text is now ' + str(mid_index))

        first_half = words[:mid_index]

        second_half = words[mid_index:length]

        sent_1 = ' '.join(first_half)

        sent_2 = ' '.join(second_half)

        sentences = [sent_1, sent_2]

        print('Splitting sentences completed.')

    else:
        first_half = words[:mid_index]

        second_half = words[mid_index:]

        sent_1 = ' '.join(first_half)

        sent_2 = ' '.join(second_half)

        sentences = [sent_1, sent_2]

        print('Splitting sentences completed.')

    return sentences


# using pipeline API for summarization task
summarization = pipeline("summarization")


# function to iterate a list and summarize elements in a list and append them to a new list
def summarize(text):
    summarized_text = []
    i = 0
    while i < len(text):
        sentences = text[i]
        summary_text = summarization(sentences, min_length=3, max_length=500)
        summarized_text.append(summary_text)
        i += 1

    new_list = []

    i = 0
    while i < len(summarized_text):
        for d in summarized_text[i]:
            new_list.append(d['summary_text'])
        i += 1

    new_text = ' '.join(map(str, new_list))

    print('Summary process finished')

    return new_text


# text summarizing pipeline containing text splitting and text summarization functions
def text_summarize_pipeline(text):
    new_text = []
    i = 0
    while i < len(text):
        print('Processing text number : ' + str(i))
        split_text = split_sentences(text[i])
        summary = summarize(split_text)
        new_text.append(summary)
        print('Finished processing text number : ' + str(i))
        i += 1

    print('Text has been summarized')

    return new_text


# summarizing text requires function to store the summarized text in memory
# therefore not all the text in the list gets summarized
# Y dataset will be spliced to only summarize a part of the list

# splicing Y dataset from element 1401 to element 2002
# Y_data = Y[1401:2002]

# feeding spliced array into text summarizing pipeline function
# Y_split = text_summarize_pipeline(Y_data)

# splicing Y dataset from element 0 to element 1401
# Y_1 = Y[0:1401]

# concatenate or join both spliced arrays into one
# Y_new = np.concatenate((Y_1, Y_split))

# Feature Extraction part
# Convert a collection of text documents to a matrix of token counts
# This implementation produces a sparse representation of the counts using scipy.sparse.csr_matrix.
# we are trying to produce a bag-of-words representation which will be displayed as a sparse matrix
# containing term frequency depending on ngram range we've set
matrix = CountVectorizer(analyzer='word',
                         token_pattern=r'\b[a-zA-Z]{3,}\b',
                         ngram_range=(1, 2), min_df=2, vocabulary=pickle.load(open("feature.pkl", "rb")))

# Uncomment if want to use summarized text for bag-of-words representation
# y = matrix.fit_transform(Y_new).toarray()

# We've already load the pre-defined vocabulary from the pickle file
# Now we can use Y dataset without summarizing the text (memory efficient)
y = matrix.transform(Y).toarray()

# Save/Dump vocabulary in a pickle file to use later during prediction
# A Mapping (e.g., a dict) where keys are terms and values are indices in the feature matrix, or an iterable over terms
pickle.dump(matrix.vocabulary_, open("feature.pkl", "wb"))

# Setting ratio for train, test and validation data
train_ratio = 0.60
validation_ratio = 0.20
test_ratio = 0.20

# train is now 60% of the entire data set
# the _junk suffix means that we drop that variable completely
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio, random_state=1)

# test is now 10% of the initial data set
# validation is now 15% of the initial data set
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio / (test_ratio + validation_ratio),
                                                random_state=1)


# Categorical encoding using sklearn's Label Encoder function
# Encode target labels with value between 0 and n_classes-1
def prepare_targets(x_train, x_test, x_val):
    le = LabelEncoder()
    le.fit(x_train)
    x_train = le.transform(x_train)
    x_test = le.transform(x_test)
    x_val = le.transform(x_val)
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    x_train_enc = to_categorical(x_train)
    x_test_enc = to_categorical(x_test)
    x_val_enc = to_categorical(x_val)
    return x_train_enc, x_test_enc, x_val_enc, le_name_mapping


# prepare output data
x_train_enc, x_test_enc, x_val_enc, le_name_mapping = prepare_targets(x_train, x_test, x_val)

input_dim = y_train.shape[1]  # Number of features

# Number of features as reference when during prediction (same input size)
print(input_dim)

# initialize Sequential model Keras
model = Sequential()

model.add(Dense(65, input_dim=input_dim, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

file_path = "./model.hdf5"

ckpt = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

early = EarlyStopping(monitor="val_loss", mode="min", patience=5)

epochs = 64
learning_rate = 0.1
decay_rate = learning_rate / epochs
momentum = 0.8
sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# fitting the model and hyper-parameters
history_0 = model.fit(y_train, x_train_enc, epochs=epochs, batch_size=40, validation_data=(y_val, x_val_enc),
                      callbacks=[ckpt, early], verbose=1)

# plot history
plt.plot(history_0.history['loss'], label='training loss')
plt.plot(history_0.history['val_loss'], label='validation loss')
plt.legend()
plt.show()

# plot history
plt.plot(history_0.history['accuracy'], label='training accuracy')
plt.plot(history_0.history['val_accuracy'], label='validation accuracy')
plt.legend()
plt.show()

# returns a compiled model
# identical to the previous one
Model = load_model('Models/model.hdf5')


# Testing part using the testing data
i = 0
predicted = []
for x_t in y_test:
    prediction = np.argmax(Model.predict(np.array([x_t])), axis=-1)

    Categories = {'Data Scientist': 0,
                  'Database Administrator': 1,
                  'Java Developer': 2,
                  'Network Administrator': 3,
                  'Project Manager': 4,
                  'Python Developer': 5,
                  'Security Analyst': 6,
                  'Software Developer': 7,
                  'Systems Administrator': 8,
                  'Web Developer': 9}

    predicted_label = list(Categories.keys())[list(Categories.values()).index(prediction[0])]
    predicted.append(predicted_label)
    print("File ->", y_test[i], "Actual label: " + x_test[i], "Predicted label: " + predicted_label)
    i += 1

# Print model evaluation using sklearn's metrics for model evaluation part
# Compute precision, recall, F-measure and support for each class.
print(precision_recall_fscore_support(x_test, predicted, average='micro'))
