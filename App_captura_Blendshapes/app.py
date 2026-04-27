
from flask import Flask, render_template, request, jsonify
import time
import os
import pandas as pd
import time

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/save_data', methods=['POST'])
def save_data():
    data_request = request.get_json()
    data = data_request['data']
    DatasetID = data_request['DatasetID'] 
    
    ID = data['Identificador']
    Edad = data['Edad']
    Sexo = data['Sexo']
    Patologia = data.get('Patologia', '') 
    Emocion = data['Emocion']
    blendshapes = data['blendshapes'] 

    timestamp = int(time.time()*1000) 

    #print(f"Almacenando datos de ID: {ID}, Emoción: {Emocion}, ID Dataset: {DatasetID}")

    data_dict = {
        'ts': [timestamp],
        'id': [ID],
        'sexo': [Sexo],
        'edad': [Edad],
        'patologia': [Patologia],
        'emocion': [Emocion],
    }

    for blendshape in blendshapes:
        data_dict[blendshape['name']] = [blendshape['score']]

    df = pd.DataFrame(data_dict)

    column_order = ['ts', 'id', 'patologia', 'sexo', 'edad'] + [blendshape['name'] for blendshape in blendshapes] + ['emocion']
    df = df[column_order]

    filename = f'{DatasetID}_dataset.csv'
    file_exists = os.path.isfile(filename)
    if file_exists:
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, mode='w', header=True, index=False)

    return jsonify({"status": "success"})


if __name__ == '__main__':
    app.run(debug=True)
