from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline. predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

#Route for HOMEPAGE

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=["GET","POST"])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            person_age=request.form.get('person_age'),
            person_income=request.form.get('person_income'),
            person_home_ownership=request.form.get('person_home_ownership'),
            person_emp_length=request.form.get('person_emp_length'),
            loan_intent=request.form.get('loan_intent'),
            loan_grade=request.form.get('loan_grade'),
            loan_amnt=request.form.get('loan_amnt'),
            loan_int_rate=request.form.get('loan_int_rate'),
            loan_percent_income=request.form.get('loan_percent_income'),
            cb_person_default_on_file=request.form.get('cb_person_default_on_file'),
            cb_person_cred_hist_length=request.form.get('cb_person_cred_hist_length')
        )
        pred_df=data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results)
    
if __name__=="__main__":
    app.run(host="0.0.0.0", port=8080)  #  debug=True) if not deployed




