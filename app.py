from __future__ import division, print_function
import os
import numpy as np

from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from tensorflow.keras.preprocessing  import image

from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'D:\\aalok_project\\flask-app-20220727T063329Z-001\\flask-app\\flask-app\\models\\model113_vgg19.h5'

# Loading trained model
model = load_model(MODEL_PATH)
model.make_predict_function()         




def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    if preds[0][0] > preds[0][1]: 
        # Printing the prediction of model.
        return ('Person is Safe. ')
    else:
        return ('Person is affected with Pneumonia.')
    print(f'Predictions: {preds}')
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        #print(preds[0][0])
        #print(preds[0][1])
        """
        if preds[0][0] > preds[0][1]:  # Printing the prediction of model.
            return('Person is Safe. ')
        else:
            return('Person is affected with Pneumonia.')
        print(f'Predictions: {preds}')
        """
    return preds



if __name__ == '__main__':
    app.run(debug=False)
