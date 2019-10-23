import requests
import os
import pandas as pd
from flask import Flask, render_template, request
import numpy as np
import dill
from sklearn.feature_extraction.text import TfidfVectorizer
from ocr_core import ocr_core
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict
from sklearn import base
from sklearn.pipeline import FeatureUnion
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from model import ColumnSelectTransformer, CorpusTransformer, DictEncoder, EstimatorTransformer
from PIL import Image
import io
from base64 import b64encode

app = Flask(__name__)

dill._dill._reverse_typemap['ClassType'] = type

model = dill.load(open('lib/models/wine_estimator.dill','rb'))

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/label', methods=['GET','POST'])
def label():
    return render_template('label.html')    

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload.html', msg='No file selected')
        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', msg='No file selected')
        if file:
            extracted_text = ocr_core(file)

            image_data = Image.open(file, 'r')
            img = np.array(image_data.getdata())
            
            file_object = io.BytesIO()
            pic = Image.fromarray(img.astype('uint8'))
            
            pic.save(file_object, 'PNG')
            #with open(file.filename, "rb") as i:
            #    encoded = b64encode(i.read())
    # create file-object in memory
            #file_object = io.BytesIO()

            #pic = Image.fromarray(img.astype('uint8'))
            
            #pic.save(file_object, 'PNG')
    # move to beginning of file so `send_file()` it will read from start    
            #file_object.seek(0)
        
            encoded = b64encode(file_object.read())
            mime = "image/jpeg"
            uri = "data:%s;base64;+%s" % (mime, encoded)
            
            return render_template('upload.html',
                                   msg='Successfully processed',
                                   extracted_text=extracted_text,
                                   img_src=uri)
        
    elif request.method == 'GET':
        return render_template('upload.html')
    
@app.route('/prediction', methods=['POST'])
def prediction():
    
    query = {}
    query['description'] = [request.form['description']]
    query['price'] = [request.form['price']]
    query['province'] = [request.form['province']]
    query['variety'] = [request.form['variety']]
    query['winery'] = [request.form['winery']]
    
    query_df = pd.DataFrame.from_dict(query, orient = 'columns')
    prediction = model.predict(query_df).item()
    
    return render_template('prediction.html', prediction = prediction)

if __name__ == '__main__':
    app.run()
