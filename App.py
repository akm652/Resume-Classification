#!/usr/bin/env python
# coding: utf-8

# In[3]:


# import libraries
import sys
import os
import glob
import re
import pickle
import numpy as np
import keras
import tempfile
import io

from io import StringIO
from pathlib import Path
from keras.utils.np_utils import to_categorical
from sklearn.feature_extraction.text import CountVectorizer
from resume_parsing import pdf_parsing, custom_preprocessing
from keras.models import load_model
from flask import Flask, request, render_template, flash, redirect, url_for, session
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# In[ ]:


# Get current directory
current_dir = Path.cwd().as_posix()

UPLOAD_FOLDER = current_dir+'/Uploads'
ALLOWED_EXTENSIONS = {'pdf'}

# # Define a flask app
app = Flask(__name__, template_folder=current_dir)
app.secret_key = "super secret key"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model saved with Keras model.save()
MODEL_PATH = 'Models/model.hdf5'

# Load your trained model
model = load_model(MODEL_PATH)


# Prediction pipeline testing using resumes inside project folder (Uncomment if want to observe prediction)
# # input resume path
# result = r'C:\Users\Akmal Faisal\Desktop\Apprentice Mini Project\deploy-mini-project\Resumes'
#
# # get directory of resume file
# new_path = os.path.abspath(result).replace("\\", "/")
# # stream = io.StringIO(result.stream.read().decode("UTF8"), newline=None)
# # result_path=result.gettempdir()
#
# # get list of entries in Resumes folder
# entries = os.listdir(new_path)
#
# Resumes = []
#
# for e in entries:
#     Resumes.append(e)
#
# predicted = []
#
# i = 0
#
# for r in Resumes:
#     resume_path = new_path + '/' + r
#
#     # Parsing the resume
#     data = pdf_parsing(resume_path)
#
#     # Preprocessing the resume into a text
#     data = custom_preprocessing(data)
#
#     # data = [i for i in sent_tokenize(data)]
#
#     # Extract Feature With CountVectorizer
#     cv = CountVectorizer(analyzer='word',
#                          token_pattern=r'\b[a-zA-Z]{3,}\b',
#                          ngram_range=(1, 2), min_df=2, vocabulary=pickle.load(open("./feature.pkl", "rb")))
#     X = cv.transform([data])
#
#     # model.predict
#     prediction = np.argmax(model.predict(X), axis=-1)
#
#     # dictionary of labels
#     Categories = {'Data Scientist': 0,
#                   'Database Administrator': 1,
#                   'Java Developer': 2,
#                   'Network Administrator': 3,
#                   'Project Manager': 4,
#                   'Python Developer': 5,
#                   'Security Analyst': 6,
#                   'Software Developer': 7,
#                   'Systems Administrator': 8,
#                   'Web Developer': 9}
#
#     result_pred = list(Categories.keys())[list(Categories.values()).index(prediction[0])]
#
#     predicted.append(result_pred)
#
#     print("Processing Resume no. : "+str(i+1))
#     print("File ->", Resumes[i], "Predicted label: " + result_pred)
#
#     if i == len(Resumes):
#         print("Completed processing "+str(len(Resumes))+" resumes")
#         break
#     else:
#         i += 1


# In[ ]:


@app.route('/')
def home():
    return render_template('resume_preds.html')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# @app.route('/', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         # check if the post request has the file part
#         if 'file' not in request.files:
#             flash('No file part')
#             return redirect(request.url)
#         file = request.files['file']
#         # If the user does not select a file, the browser submits an
#         # empty file without a filename.
#         if file.filename == '':
#             flash('No selected file')
#             return redirect(request.url)
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#             return redirect(url_for('predict', name=filename))
#     return render_template('resume_preds.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        result = request.files['myFile']

        # get directory of resume file
        new_path = os.path.abspath(result).replace("\\", "/")
        # stream = io.StringIO(result.stream.read().decode("UTF8"), newline=None)
        # result_path=result.gettempdir()
        # Parsing the resume
        data = pdf_parsing(new_path)

        # Preprocessing the resume into a text
        data = custom_preprocessing(data)

        # data = [i for i in sent_tokenize(data)]

        # Extract Feature With CountVectorizer
        cv = CountVectorizer(analyzer='word',
                             token_pattern=r'\b[a-zA-Z]{3,}\b',
                             ngram_range=(1, 2), min_df=2, vocabulary=pickle.load(open("feature.pkl", "rb")))
        X = cv.transform([data])

        # model.predict
        prediction = np.argmax(model.predict(X), axis=-1)

        # dictionary of labels
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

        result_pred = list(Categories.keys())[list(Categories.values()).index(prediction[0])]

    return render_template("resume_result.html", result=result_pred)


if __name__ == "__main__":
    app.debug = True
    app.run()
