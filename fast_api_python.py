from fastapi import FastAPI
from pydantic import BaseModel 
import pickle 
import json



app =FastAPI()

class model_input(BaseModel):
    Pregnancies :int 
    Glucose : int
    BloodPressure :int 
    SkinThickness :int 
    Insulin : int 
    BMI :float
    DiabetesPedigreeFunction :float 
    Age :int

# loading the saved model 
model = pickle.load(open('logistic_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav','rb'))

@app.post('/prediction')
def pred(input_parameters : model_input):
    input_data = input_parameters.json()
    input_dict = json.loads(input_data)

    Pregnancies = input_dict['Pregnancies']
    Glucose = input_dict['Glucose']
    BloodPressure = input_dict['BloodPressure']
    SkinThickness = input_dict['SkinThickness']
    Insulin = input_dict['Insulin']
    BMI = input_dict['BMI']
    DiabetesPedigreeFunction = input_dict['DiabetesPedigreeFunction']
    Age = input_dict['Age']

    input_list = [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,BMI, DiabetesPedigreeFunction, Age]]

    input_val = scaler.transform(input_list)
    prediction = model.predict(input_val)[0]

    if prediction==1:
        return "Positive"
    else:
        return 'Negative'

