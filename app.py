from flask import Flask, request, jsonify
from joblib import load
from flask_cors import CORS
import pandas as pd
app = Flask(__name__)
CORS(app) 

model = load('model1.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    data_dict = {
      'chieuDai': [data['chieuDai']],
      'chieuNgang': [data['chieuNgang']],
      'dienTich': [data['dienTich']],
      'Phongngu': [data['Phongngu']],
      'SoTang': [data['SoTang']],
      'PhongTam': [data['PhongTam']],
      'Loai': [data['Loai']],
      'GiayTo': [data['GiayTo']],
      'TinhTrangNoiThat': [data['TinhTrangNoiThat']],
      'Phuong': [data['Phuong']]
    }
    print(data_dict)
    features = pd.DataFrame(data_dict)
    print(features)
    prediction = model.predict(features)[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
