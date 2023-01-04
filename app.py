import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
app=Flask(__name__)
md=pickle.load(open("regmodel.pkl","rb"))
sd=pickle.load(open('Scaler.pkl','rb'))
@app.route('/')

def home():
    return render_template('UI.html')
@app.route('/Predict_api',methods=['POST'])

def Predict_api():
    data=request.json['data']
    print("You entered this data ",data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=sd.transform(np.array(list(data.values())).reshape(1,-1))
    pred=md.predict(new_data)
    print(pred[0])
    return jsonify(pred[0])
@app.route( '/Predict',methods=['POST'] )
def Predict():
    data=[float(x) for x in request.form.values()]
    final_input=sd.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=md.predict(final_input)[0]
    output2=round(output,3)
    return render_template('UI.html',pred_text="We Predict the price will be {}".format(output2))

if __name__=="__main__":
    app.run(debug=True)