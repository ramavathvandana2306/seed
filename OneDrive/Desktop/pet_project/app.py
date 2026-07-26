from flask import Flask, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

model = tf.keras.models.load_model("model.h5")

@app.route('/')

def home():
    return '''
    <h1>Cats vs Dogs Classifier</h1>

    <form method="POST" action="/predict" enctype="multipart/form-data">

        <input type="file" name="file">

        <input type="submit">

    </form>
    '''

@app.route('/predict', methods=['POST'])

def predict():

    file = request.files['file']

    file.save("temp.jpg")

    img = image.load_img("temp.jpg", target_size=(224,224))

    img = image.img_to_array(img)
    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    result = "Dog" if prediction[0][0] > 0.5 else "Cat"

    return f"<h1>Prediction: {result}</h1>"

app.run(debug=True)
