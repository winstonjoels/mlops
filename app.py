#!/usr/bin/env python
# coding: utf-8

# In[4]:


from flask import Flask,request, url_for, redirect, render_template, jsonify
from pycaret.regression import *
import pandas as pd
import pickle
import numpy as np
import random
import csv

def randN():
    N=7
    min = pow(10, N-1)
    max = pow(10, N) - 1
    id = random.randint(min, max)
    return id

app = Flask(__name__)

global model, cols, id

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/eval',methods=['POST'])
def eval():
    filename = "data/Details.csv"
    fields = [] 
    rows = [] 
    with open(filename, 'r') as csvfile: 
        csvreader = csv.reader(csvfile) 
        fields = next(csvreader) 
        for row in csvreader: 
            rows.append(row) 
    df = pd.DataFrame(rows, columns = ['ID', 'Name', 'Predicted', 'Actual'])
    return render_template("eval.html", tables=df, titles=df.columns.values)

@app.route('/model/<name>',methods=['POST'])
def model(name):
    global model, cols, id
    id = randN()
    cols = ['age','job','marital','education']
    file = name+".html"
    return render_template(file)

@app.route('/predict/<name>',methods=['POST'])
def predict(name):
    global model, cols
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    name1 = name+"_training_pipeline"
    model = load_model(name1)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction = model.predict(data_unseen)
    if int(prediction)==1:
        pred="Eligible for loan"
    else:
        pred="Not eligible for loan"
    
    actual = '?'
    list = [id, name, int(prediction), actual]
    list2 = ['ID','Name', 'Predicted', 'Actual']
    df = np.array(list)
    df = pd.DataFrame([df], columns=list2)
    
    df.to_csv('data/Details.csv', mode='a', header=False, index=False)
    
    file = name+".html"
    return render_template(file,pred='{}'.format(pred))

if __name__ == '__main__':
    app.run()


# In[ ]:




