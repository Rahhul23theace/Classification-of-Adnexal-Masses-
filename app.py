



from flask import Flask, render_template, request
import numpy as np
import pickle

# Load your trained model
with open('pickles/best_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get user input
    features = [float(x) for x in request.form.values()]

    # Convert user input into a numpy array and reshape
    input_features = np.array(features).reshape(1, -1)

    # Make a prediction using the loaded model
    prediction = model.predict(input_features)

    # Return the prediction result
    return render_template('index.html', prediction_text='Predicted value: {}'.format(prediction[0]))


if __name__ == "__main__":
    app.run(debug=True)
