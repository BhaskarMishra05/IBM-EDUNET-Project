import os
import sys
import pandas as pd
from flask import Flask, request, render_template
from src.utils import load_obj
from src.pipeline.prediction_pipeline import MAIN, DATA 

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoints():
    if request.method == 'GET':
        return render_template('home.html')
    else:

        data = DATA(
            age=request.form.get('age'),
            workclass=request.form.get('workclass'),
            fnlwgt=request.form.get('fnlwgt'),
            education=request.form.get('education'),
            educational_num=request.form.get('educational_num'),
            marital_status=request.form.get('marital_status'),
            occupation=request.form.get('occupation'),
            relationship=request.form.get('relationship'),
            race=request.form.get('race'),
            gender=request.form.get('gender'),
            capital_gain=request.form.get('capital_gain'),
            capital_loss=request.form.get('capital_loss'),
            hours_per_week=request.form.get('hours_per_week'),
            native_country=request.form.get('native_country')
        )

        pred_data = data.data_frame()

        model = MAIN()
        results = model.predict(pred_data)

        return render_template('home.html', results=results[0])

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
