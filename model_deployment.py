from flask import Flask, request, render_template
import pickle
import numpy as np

app= Flask(__name__)
model = pickle.load(open('model.sav', 'rb'))


@app.route('/')
def home():
    print('start')
    return render_template('predict.html')


@app.route('/predict', methods=['POST'])
def predict():
    input_values = [float(x) for x in request.form.values()]
    features = [np.array(input_values)]
    output = model.predict(features)
    print('predict')
    return render_template('predict.html', prediction='Species should be {}'.format(output[0].upper()))


if __name__ == '__main__':
    app.run(port=5000, debug=True)