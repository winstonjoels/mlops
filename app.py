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
from sklearn.metrics import accuracy_score

def randN():
    N=7
    min = pow(10, N-1)
    max = pow(10, N) - 1
    id = random.randint(min, max)
    return id

app = Flask(__name__)

global model, cols, id, predict

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
        for row in csvreader: 
            rows.append(row) 
    df = pd.DataFrame(rows, columns = ['ID', 'Name', 'Predicted', 'Actual'])
    df = df.loc[df['Actual'] == '?']
    return render_template("eval.html", column_names=df.columns.values, row_data=list(df.values.tolist()), link_column="Actual", zip=zip)

@app.route('/after_store',methods=['POST'])
def after_eval():
    columns1 = ['ID', 'Name', 'Predicted', 'Actual']
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = columns1)
    
    output = []
    filename = "data/Details.csv"
    with open(filename, 'r') as csvfile: 
        csvreader = csv.reader(csvfile) 
        for line in csvreader:
            if str(data_unseen['ID'][0]) == line[0]:
                line[3] = str(data_unseen['Actual'][0])
            output.append(line)
        list2 = ['ID','Name', 'Predicted', 'Actual']
        df = pd.DataFrame(output, columns=list2)
        df.to_csv('data/Details.csv', mode='w', header=False, index=False)
    return eval()

@app.route('/model/<name>',methods=['GET','POST'])
def model(name):
    global model, cols, id
    id = randN()
    if(name=="bank"):
        cols = ['age','job','marital','education']
    else:
        cols = ['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction','high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking']
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
    if name=="bank":
        if int(prediction)==1:
            pred="Eligible for loan"
        else:
            pred="Not eligible for loan"
    else:
        if int(prediction)==1:
            pred="critical condition"
        else:
            pred="normal condition"
    
    actual = '?'
    list = [id, name, int(prediction), actual]
    list2 = ['ID','Name', 'Predicted', 'Actual']
    df = np.array(list)
    df = pd.DataFrame([df], columns=list2)
    
    df.to_csv('data/Details.csv', mode='a', header=False, index=False)
    
    file = name+".html"
    return render_template(file, id=id, pred='{}'.format(pred))

@app.route('/model_eval',methods=['POST'])
def model_eval():
    filename = "data/Details.csv"
    name = ['bank']
    rows = []
    accuracy = []
    with open(filename, 'r') as csvfile: 
        csvreader = csv.reader(csvfile) 
        for row in csvreader: 
            rows.append(row)
    
    df = np.array(rows)
    df = pd.DataFrame(df, columns = ['ID', 'Name', 'Predicted', 'Actual'])
    acc = "Not_enough_data"
    accuracy = {}
    for n in name:
        df1 = df.loc[df['Name'] == n].loc[df['Actual'] != '?']
        y_true = df1['Actual'].to_list()
        y_pred = df1['Predicted'].to_list()
        if len(y_true)>=20 and len(y_pred)>=20:
            acc = str(accuracy_score(y_true, y_pred)*100)
        accuracy[n] = acc
        
    return render_template("model_eval.html", accuracy = accuracy, name = name)

if __name__ == '__main__':
    app.run()


# In[ ]:




