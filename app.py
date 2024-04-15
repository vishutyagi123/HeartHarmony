from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
import pickle
import joblib
from joblib import load

app = Flask(__name__)


with open('.venv\\src\\xgb_classifier.pkl', 'rb') as file:
    xgb_classifier = pickle.load(file)

with open('.venv\src\State_dic.pkl', 'rb') as file:
    State_dic = pickle.load(file)

with open('.venv\src\GeneralHealth_dic.pkl', 'rb') as file:
    GeneralHealth_dic = pickle.load(file)

with open('.venv\src\LastCheckupTime_dic.pkl', 'rb') as file:
    LastCheckupTime_dic = pickle.load(file)

with open('.venv\src\RemovedTeeth_dic.pkl', 'rb') as file:
    RemovedTeeth_dic = pickle.load(file)

with open('.venv\src\HadDiabetes_dic.pkl', 'rb') as file:
    HadDiabetes_dic = pickle.load(file)

with open('.venv\src\SmokerStatus_dic.pkl', 'rb') as file:
    SmokerStatus_dic = pickle.load(file)

with open('.venv\src\ECigaretteUsage_dic.pkl', 'rb') as file:
    ECigaretteUsage_dic = pickle.load(file)

with open('.venv\src\AgeCategory_dic.pkl', 'rb') as file:
    AgeCategory_dic = pickle.load(file)

with open('.venv\src\RaceEthnicityCategory_dic.pkl', 'rb') as file:
    RaceEthnicityCategory_dic = pickle.load(file)

with open('.venv\src\TetanusLast10Tdap_dic.pkl', 'rb') as file:
    TetanusLast10Tdap_dic = pickle.load(file)

with open('.venv\src\AgeCategory_dic.pkl', 'rb') as file:
    AgeCategory_dic = pickle.load(file)

with open('.venv\src\CovidPos_dic.pkl', 'rb') as file:
    CovidPos_dic = pickle.load(file)


def process_data(df):

    col = ['PhysicalActivities','HadAngina', 'HadStroke', 'HadAsthma', 'HadSkinCancer', 'HadCOPD', 'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis', 'DeafOrHardOfHearing', 'BlindOrVisionDifficulty', 'DifficultyConcentrating',  'DifficultyWalking', 'DifficultyDressingBathing', 'DifficultyErrands', 'ChestScan', 'AlcoholDrinkers', 'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver', 'HighRiskLastYear']

    def binary_map(x):
        return x.map({'Yes': 1, 'No': 0})

    df[col] = df[col].apply(binary_map)
    df['Sex'] = df['Sex'].map({'Male': 1, 'Female': 0})

    categ = ['State', 'GeneralHealth', 'LastCheckupTime', 'RemovedTeeth', 'HadDiabetes', 'SmokerStatus',
             'ECigaretteUsage', 'AgeCategory', 'RaceEthnicityCategory', 'TetanusLast10Tdap', 'CovidPos']

    for cat in categ:
        i = f'{cat}_dic'
        reference = globals()[i]
        df[cat] = df[cat].map(reference)

    return df


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    dic = {}

    dic['State'] = request.form.get('State')
    dic['Sex'] = request.form.get('Sex')
    dic['GeneralHealth'] = request.form.get('GeneralHealth')
    dic['PhysicalHealthDays'] = request.form.get('PhysicalHealthDays')
    dic['MentalHealthDays'] = request.form.get('MentalHealthDays')
    dic['LastCheckupTime'] = request.form.get('LastCheckupTime')
    dic['PhysicalActivities'] = request.form.get('PhysicalActivities')
    dic['SleepHours'] = request.form.get('SleepHours')
    dic['RemovedTeeth'] = request.form.get('RemovedTeeth')
    dic['HadAngina'] = request.form.get('HadAngina')
    dic['HadStroke'] = request.form.get('HadStroke')
    dic['HadAsthma'] = request.form.get('HadAsthma')
    dic['HadSkinCancer'] = request.form.get('HadSkinCancer')
    dic['HadCOPD'] = request.form.get('HadCOPD')
    dic['HadDepressiveDisorder'] = request.form.get('HadDepressiveDisorder')
    dic['HadKidneyDisease'] = request.form.get('HadKidneyDisease')
    dic['HadArthritis'] = request.form.get('HadArthritis')
    dic['HadDiabetes'] = request.form.get('HadDiabetes')
    dic['DeafOrHardOfHearing'] = request.form.get('DeafOrHardOfHearing')
    dic['BlindOrVisionDifficulty'] = request.form.get('BlindOrVisionDifficulty')
    dic['DifficultyConcentrating'] = request.form.get('DifficultyConcentrating')
    dic['DifficultyWalking'] = request.form.get('DifficultyWalking')
    dic['DifficultyDressingBathing'] = request.form.get('DifficultyDressingBathing')
    dic['DifficultyErrands'] = request.form.get('DifficultyErrands')
    dic['SmokerStatus'] = request.form.get('SmokerStatus')
    dic['ECigaretteUsage'] = request.form.get('ECigaretteUsage')
    dic['ChestScan'] = request.form.get('ChestScan')
    dic['RaceEthnicityCategory'] = request.form.get('RaceEthnicityCategory')
    dic['AgeCategory'] = request.form.get('AgeCategory')
    dic['HeightInMeters'] = request.form.get('HeightInMeters')
    dic['WeightInKilograms'] = request.form.get('WeightInKilograms')
    BMI = int(dic['WeightInKilograms']) / (int(dic['HeightInMeters']) * int(dic['HeightInMeters']))
    dic['BMI'] = BMI
    dic['AlcoholDrinkers'] = request.form.get('AlcoholDrinkers')
    dic['HIVTesting'] = request.form.get('HIVTesting')
    dic['FluVaxLast12'] = request.form.get('FluVaxLast12')
    dic['PneumoVaxEver'] = request.form.get('PneumoVaxEver')
    dic['TetanusLast10Tdap'] = request.form.get('TetanusLast10Tdap')
    dic['HighRiskLastYear'] = request.form.get('HighRiskLastYear')
    dic['CovidPos'] = request.form.get('CovidPos')


    print(dic)
    print(f'Size of dic is : {len(dic)}')
    # creating one dataframe
    df = pd.DataFrame(dic, index=[0])
    df = process_data(df)

    for i in df.columns:
        df[i] = pd.to_numeric(df[i])

    print(f'DataFrame is : {df}')
    pred = xgb_classifier.predict(df)
    print(f"Model Prediction is : {pred}")
    if pred:
        pred = "Your risk of heart attack is high. Please consult a healthcare professional immediately."
    else:
        pred = "Your risk of heart attack is low. Keep up with healthy habits and regular check-ups to maintain heart health."
    return pred

@app.route('/show_results')
def show_results():
    print("Entering results function")
    prediction = request.args.get('prediction')
    print(f'Requested predictions are : {prediction}')
    return render_template('resulting_page.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
