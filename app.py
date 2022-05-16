from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('modeld.pkl', 'rb'))


@app.route('/')
def hello_world():
    return render_template("index_new.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    new_float = float_features.copy()
    new_float[0] = (new_float[0] + 500) / 4500 #rev
    new_float[1] /= 20000 #amt
    new_float[2] /= 120 #cnt
    new_float[3] /= 2 #4-1

    final = [np.array(new_float)]
    # final = [[0.790624, 0.116687, 0.428571, 0.593315]]
    print(float_features)
    print(final)
    prediction = model.predict(final)
    prediction = float(prediction)
    print(prediction)
    # return prediction

    if prediction == 0:
        return render_template('index_new.html',
                               pred='For total trasaction amount "{1}", total trasaction count "{2}", total count '
                                    'change Q4 to Q1 "{3}", total revolving balance "{0}" customer is going to '
                                    'chrun'.format(float_features[0], float_features[1], float_features[2],
                                                   float_features[3], prediction))
    else:
        return render_template('index_new.html',
                               pred='For total trasaction amount "{1}", total trasaction count "{2}", total count '
                                    'change Q4 to Q1 "{3}", total revolving balance "{0}" customer is not going to '
                                    'chrun'.format(float_features[0], float_features[1], float_features[2],
                                                   float_features[3], prediction))



if __name__ == '__main__':
    app.run(debug=True)
