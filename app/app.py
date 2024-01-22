from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('../algorithms/model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the HTML form
        rooms = int(request.form['rooms'])
        area = float(request.form['area'])
        floor = int(request.form['floor'])
        location = int(request.form['location'])

        # Create a DataFrame from the input data
        input_data = pd.DataFrame({'rooms': [rooms], 'area': [area], 'floor': [floor], 'location': [location]})

        # Make predictions using your model
        prediction = model.predict(input_data)

        # Display the prediction on a new page
        return render_template('result.html', prediction=prediction[0])

    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
