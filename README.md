# Loan Approval Prediction - Sparsh Jain

### Introduction About the Data :

**The dataset** The objective is to develop a classification model to predict whether a loan application, based on provided attributes, will be approved or denied.

There are 10 independent variables (including `id`):

* `id` : unique identifier of each entry
* `person_age` : Age of person
* `person_income` : Income of a person
* `person_home_ownership` : Status of home ownership i.e. "RENT", "OWN", "MORTGAGE" and "OTHER"
* `person_emp_length` : Employment length in years
* `loan_intent` : Purpose for taking loan
* `loan_grade` : Loan Grade
* `loan_amount` : Amount of the loan
* `loan_int_rate` : Interest rate for the loan
* `loan_percent_income` : The percent of the loan to income in percentage(%)
* `cb_person_default_on_file` : The presence of historical default (Y or N)
* `cb_person_cred_hist_length` : Credit history length

Target variable:
* `loan_status`: Loan status (0 is no; 1 is yes)

Dataset Source Link :
https://www.kaggle.com/datasets/chilledwanker/loan-approval-prediction

# Approach for the project 

1. Data Ingestion : 
    * In Data Ingestion phase the data is first read as csv. 
    * Then the data is split into training and testing and saved as csv file.

2. Data Transformation : 
    * In this phase a ColumnTransformer Pipeline is created.
    * for Numeric Variables first SimpleImputer is applied with strategy median , then Standard Scaling is performed on numeric data.
    * for Categorical Variables SimpleImputer is applied with most frequent strategy, then ordinal encoding performed , after this data is scaled with Standard Scaler.
    * This preprocessor is saved as pickle file.

3. Model Training : 
    * In this phase base model is tested . The best model found was catboost regressor.
    * After this hyperparameter tuning is performed on catboost and knn model.
    * A final VotingRegressor is created which will combine prediction of catboost, xgboost and knn models.
    * This model is saved as pickle file.

4. Prediction Pipeline : 
    * This pipeline converts given data into dataframe and has various functions to load pickle files and predict the final results in python.

5. Flask App creation : 
    * Flask app is created with User Interface to predict the gemstone prices inside a Web Application.

# Exploratory Data Analysis Notebook

Link : [EDA Notebook](./notebook/1_EDA_Gemstone_price.ipynb)

# Model Training Approach Notebook

Link : [Model Training Notebook](./notebook/2_Model_Training_Gemstone.ipynb)

# Model Interpretation with LIME 

Link : [LIME Interpretation](./notebook/3_Explainability_with_LIME.ipynb)
