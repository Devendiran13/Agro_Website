from flask import Flask, render_template,request
import pickle
import numpy as np
from Adafruit_IO import Client, Data
import pandas as pd
plant_recommendation_model_path = 'models/nAIVE_BAYESplantrec.pkl'
plant_recommendation_model = pickle.load(
    open(plant_recommendation_model_path, 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('plant-index.html')

@ app.route('/plant-predict', methods=['POST'])
def plant_prediction():
    title = 'Plant Prediction'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        Temperature=int(request.form['Temperature'])
        Humidity=int(request.form['Humidity'])
        ph=int(request.form['ph'])
        Rainfall=int(request.form['rainfall'])

        data = np.array([[N, P, K,Temperature,Humidity,ph,Rainfall]])
        my_prediction = plant_recommendation_model.predict(data)
        final_prediction = my_prediction[0]
        return render_template('plant-result.html', prediction=final_prediction, title=title)

if __name__ == '__main__':
    app.run(debug=True)