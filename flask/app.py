from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
import os
from keras.preprocessing import image

app = Flask('__name__')


def predictions(img_file):
    loaded_model = tf.keras.models.load_model('cat_v_dog.h5')
    test_image = image.load_img(img_file, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = loaded_model.predict(test_image)
    if result[0][0] == 1:
        os.remove(img_file)
        return 'dog'
    elif [0][0] == 0:
        os.remove(img_file)
        return 'cat'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save(uploaded_file.filename)
        prediction = predictions(uploaded_file.filename)
        return render_template('index.html', prediction=prediction)
    else:
        return render_template('index.html', message='Something went wrong!')


if __name__ == '__main__':
    app.run(debug=False)
