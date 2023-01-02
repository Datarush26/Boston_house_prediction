import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
md=pickle.load(open("regmodel.pkl","rb"))
sd=pickle.load(open('Scaler.pkl','rb'))
@app.route('/')

def home():
    return render_template('UI.html')
@app.route('/Predict',methods=['POST'])

def Predict():
    data=request.json['data']
    print("You entered this data ",data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=sd.transform(np.array(list(data.values())).reshape(1,-1))
    pred=md.predict(new_data)
    print(pred[0])
    return jsonify(pred[0])


if __name__=="__main__":
    app.run(debug=True)