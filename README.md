# Loan Approval Prediction - Sparsh Jain

### Introduction About the Data :

**The dataset** The objective is to develop a classification model to predict whether a loan application, based on provided attributes, will be approved or denied.

There are 12 independent variables (including `id`):

* `id` : unique identifier of each entry
* `person_age` : Age of person
* `person_income` : Income of a person
* `person_home_ownership` : Status of home ownership i.e. "RENT", "OWN", "MORTGAGE" and "OTHER"
* `person_emp_length` : Employment length in years
* `loan_intent` : Purpose for taking a loan
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
    * Read the Data: The raw data is loaded from a CSV file containing loan application details. 
    * **Train-Test Split**: The dataset is then split into training and testing sets using a typical 80-20 split ratio. This split ensures that we have enough data to train the model and also evaluate its performance on unseen data. The splits are saved as separate CSV files for future use.

2. Data Transformation : 
    * **ColumnTransformer Pipeline**: A ColumnTransformer pipeline is created to handle different data types efficiently.
    * **Handling Numeric Variables**:
      * **Imputation**:For numeric columns with missing values, SimpleImputer is used with the median strategy to fill in the missing values. 
      * **Scaling**: After imputation, the numeric variables are scaled using StandardScaler to standardize the features, ensuring they have a mean of 0 and standard deviation of 1. This helps improve the performance of many machine learning models.
   * **Handling Categorical Variables**:
      * **Imputation**:For categorical columns with missing values, SimpleImputer is applied with the most frequent strategy, which fills in missing entries with the mode (most common) value.
      * **Encoding**:The categorical variables are then transformed using Ordinal Encoding, which converts categorical values into integer labels.
      * **Scaling**: Finally, categorical features are also scaled using StandardScaler to ensure uniformity in how features are treated by the model.
   * **Saving Preprocessor**: Once the preprocessor pipeline is set up, it is saved as a pickle file to reuse during model prediction and deployment phases.

3. Model Training:
   * **Initial Model Testing**: The first step in model training involves evaluating a variety of machine learning models. After testing several classifiers, the CatBoost classifier was identified as the best-performing model for this task.
   * **Hyperparameter Tuning**: Once the base CatBoost model was selected, we perform hyperparameter tuning using techniques such as Grid Search or Random Search. This helps find the optimal parameters for the model to enhance its performance.
   * **Model Saving**: After fine-tuning, the final model is saved as a pickle file for later use in predictions.

4. Prediction Pipeline:
   * **Data Conversion**: The prediction pipeline is designed to convert incoming data into a format suitable for the model. It takes the raw input data, processes it (using the preprocessor pipeline saved earlier), and formats it into a pandas DataFrame.
   * **Loading Model and Preprocessor**: The saved pickle files (model and preprocessor) are loaded into the pipeline. The data is transformed using the preprocessor, and the prediction is made using the trained model.
   *  **Result Prediction**: The pipeline outputs the predicted loan approval status (approved/denied) based on the processed input data.

5. Flask App creation : 
    * **Web Application**: A Flask web application is created to allow users to interact with the model via a simple user interface (UI). The UI prompts users to input loan application details (e.g., income, loan amount, credit score, etc.).
    * **Prediction Interaction**: When a user submits their information, the web app calls the prediction pipeline to process the data, runs the trained model, and displays the predicted loan approval status on the web page.
    * **Deployment**: The Flask app is deployed, making it accessible to users via a web browser, allowing them to easily check the loan approval status in real time.
