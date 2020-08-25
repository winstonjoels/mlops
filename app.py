#!/usr/bin/env python
# coding: utf-8

# In[4]:


from flask import Flask,request, url_for, redirect, render_template, jsonify
from pycaret.regression import *
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

global model, cols

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/model/<name>',methods=['POST'])
def model(name):
    global model, cols
    cols = ['age','job','marital','education']
    file = name+".html"
    return render_template(file)

@app.route('/predict/<name>',methods=['POST'])
def predict(name):
    global model
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    name1 = name+"_training_pipeline"
    model = load_model(name1)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction = model.predict(data_unseen)
    if int(prediction)==1:
        pred="Eligible for Loan"
    else:
        pred="Not eligible for Loan"
    file = name+".html"
    return render_template(file,pred='{}'.format(pred))

if __name__ == '__main__':
    app.run()


# In[ ]:




