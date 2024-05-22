import pickle
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

# Load the dataset
data = {
    'age': [75.0, 55.0, 65.0, 50.0, 65.0],
    'anaemia': [0, 0, 0, 1, 1],
    'creatinine_phosphokinase': [582, 7861, 146, 111, 160],
    'diabetes': [0, 0, 0, 0, 1],
    'ejection_fraction': [20, 38, 20, 20, 20],
    'high_blood_pressure': [1, 0, 0, 0, 0],
    'platelets': [265000.00, 263358.03, 162000.00, 210000.00, 327000.00],
    'serum_creatinine': [1.9, 1.1, 1.3, 1.9, 2.7],
    'serum_sodium': [130, 136, 129, 137, 116],
    'sex': [1, 1, 1, 1, 0],
    'smoking': [0, 0, 1, 0, 0],
    'time': [4, 6, 7, 7, 8],
    'DEATH_EVENT': [1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    # Handle the query
    feature = request.form.get('feature')
    plot_url = visualize_data(feature)
    return jsonify({'plot_url': plot_url})

def visualize_data(feature):
    # Generate a plot for the specified feature
    plt.figure(figsize=(10, 6))
    sns.histplot(df[feature], kde=True, color='blue')
    plt.title(f'Distribution of {feature.capitalize()}')
    plt.xlabel(feature.capitalize())
    plt.ylabel('Frequency')

    # Save plot to a string buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return f'data:image/png;base64,{img_str}'

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    final_features = scaler.transform(final_features)    
    prediction = model.predict(final_features)
    print("final features",final_features)
    print("prediction:",prediction)
    output = round(prediction[0], 2)
    print(output)

    if output == 0:
        return render_template('index.html', prediction_text='THE PATIENT IS NOT LIKELY TO HAVE A HEART FAILURE')
    else:
         return render_template('index.html', prediction_text='THE PATIENT IS LIKELY TO HAVE A HEART FAILURE')
        
@app.route('/predict_api',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
