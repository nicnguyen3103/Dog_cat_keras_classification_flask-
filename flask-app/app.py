import os
from flask import Flask, render_template, request
from flask import send_from_directory
from keras.models import load_model
# from keras.preprocessing import image
import numpy as np
import tensorflow as tf

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = './static/'
MODEL_NAME = 'catdog_classifer.h5'
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

model_new = tf.keras.models.load_model(STATIC_FOLDER + MODEL_NAME)

folder= './uploads/481.jpg'
def api(full_path):

    img = tf.io.read_file(full_path)
    image = tf.image.decode_jpeg(img, channels=3)
    image = tf.image.resize(image, [IMAGE_WIDTH, IMAGE_HEIGHT])
    image /= 255.0
    image= np.expand_dims(image, axis=0)
    prediction = model_new.predict(image)
    
    return prediction

# home page
@app.route('/')
def home():
   return render_template('index.html')


# procesing uploaded file and predict it
@app.route('/upload', methods=['POST','GET'])
def upload_file():

    if request.method == 'GET':
        return render_template('index.html')
    else:
        file = request.files['image']
        full_name = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(full_name)

        indices = {0: 'Cat', 1: 'Dog', 2: 'Invasive carcinomar', 3: 'Normal'}
        result = api(full_name)

        predicted_class = np.asscalar(np.argmax(result, axis=1))
        accuracy = round(result[0][predicted_class] * 100, 2)
        label = indices[predicted_class]

    return render_template('predict.html', image_file_name = file.filename, label = label, accuracy = accuracy)


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.debug = True
    app.run(debug=True)
    app.debug = True
