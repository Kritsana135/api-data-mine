import pickle
import numpy as np

from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from flask_cors import CORS, cross_origin

app = Flask(__name__)
model = pickle.load(open('model/mobile_phone_model.pkl', 'rb'))
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    payment = request.form['payment']
    timespend = request.form['timespend']
    unlock = request.form['unlock']
    size = request.form['size']
    age = request.form['age']
    gender = request.form['gender']
    salary = request.form['salary']

    tmp_str = str(payment) + str(timespend) + str(unlock) + str(size) + str(age) + str(gender) + str(salary)

    #Check if input is empty or not
    #Check if enough features or not
    if not tmp_str or len(tmp_str) != 26:
        return jsonify(result="required more information",status="faile")

    else:
        tmp_str_split = list(tmp_str)

        tmp_int = []
        for i in tmp_str_split:
            tmp_int.append(int(i))

        final_features = np.array(tmp_int)
        final_input = final_features.reshape(1,-1)
        prediction = model.predict(final_input)
        output = prediction[0]

        return jsonify(result=output,status="success")


# if __name__ == '__main__':
#     app.run(debug=True)