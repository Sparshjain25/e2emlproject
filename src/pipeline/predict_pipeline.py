import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path = os.path.join(os.getcwd(), "artifacts", "model.pkl")
            print("model_path ",model_path)
            preprocessor_path=os.path.join(os.getcwd(), "artifacts", "preprocessor.pkl")
            model=load_object(file_path=model_path)
            print(" pp ", preprocessor_path)
            preprocessor=load_object(file_path=preprocessor_path)
            # print("FEATURE________________________________________________\n",features)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            if preds[0]:
                return "Approved!"
            else:
                return "Rejected!!"
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self, 
                 person_age: int, 
                 person_income: int, 
                 person_home_ownership: str, 
                 person_emp_length, 
                 loan_intent: str, 
                 loan_grade: str, 
                 loan_amnt: int, 
                 loan_int_rate, 
                 loan_percent_income, 
                 cb_person_default_on_file: str, 
                 cb_person_cred_hist_length: int):
        self.person_age=person_age
        self.person_income=person_income
        self.person_home_ownership=person_home_ownership
        self.person_emp_length=person_emp_length
        self.loan_intent=loan_intent
        self.loan_grade=loan_grade
        self.loan_amnt=loan_amnt
        self.loan_int_rate=loan_int_rate
        self.loan_percent_income=loan_percent_income
        self.cb_person_default_on_file=cb_person_default_on_file
        self.cb_person_cred_hist_length=cb_person_cred_hist_length
    
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict={
                "person_age":[self.person_age],
                "person_income":[self.person_income],
                "person_home_ownership":[self.person_home_ownership],
                "person_emp_length":[self.person_emp_length],
                "loan_intent":[self.loan_intent],
                "loan_grade":[self.loan_grade],
                "loan_amnt":[self.loan_amnt],
                "loan_int_rate":[self.loan_int_rate],
                "loan_percent_income":[self.loan_percent_income],
                "cb_person_default_on_file":[self.cb_person_default_on_file],
                "cb_person_cred_hist_length":[self.cb_person_cred_hist_length],
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)


