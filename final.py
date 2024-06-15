from flask import Flask, request, jsonify
import pickle
import numpy as np
import os
import tempfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__)

def load_model():
    with open('model1.pkl', 'rb') as f:
        model1 = pickle.load(f)
    with open('model2.pkl', 'rb') as f:
        model2 = pickle.load(f)
    with open('model3.pkl', 'rb') as f:
        model3 = pickle.load(f)
    return model1, model2, model3

model1, model2, model3 = load_model()

def preprocessing(path):
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        path,
        target_size=(150, 150),
        class_mode=None,
        batch_size=1,
        shuffle=False
    )
    return test_generator 

def get_predictions(model, generator):
    steps = generator.n // generator.batch_size + 1  # Make sure to include the remainder
    predictions = model.predict(generator, steps=steps, verbose=1)
    return predictions[:generator.n]  # Trim to the actual number of samples

def majority_voting(preds1, preds2, preds3):
    binary_preds_test_model1 = (preds1 > 0.5).astype(int)
    binary_preds_test_model2 = (preds2 > 0.5).astype(int)
    binary_preds_test_model3 = (preds3 > 0.5).astype(int)

    X_test_preds = np.column_stack((binary_preds_test_model1, binary_preds_test_model2, binary_preds_test_model3))

    final_preds_test = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=X_test_preds)
    return final_preds_test

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        with tempfile.TemporaryDirectory() as tmpdirname:
            file_path = os.path.join(tmpdirname, file.filename)
            file.save(file_path)

            # Ensure the temporary directory structure matches what flow_from_directory expects
            tmpdir_for_generator = os.path.join(tmpdirname, 'uploads')
            os.makedirs(tmpdir_for_generator, exist_ok=True)
            os.rename(file_path, os.path.join(tmpdir_for_generator, file.filename))

            test_generator = preprocessing(tmpdirname)

            preds_test_model1 = get_predictions(model1, test_generator)
            preds_test_model2 = get_predictions(model2, test_generator)
            preds_test_model3 = get_predictions(model3, test_generator)

            final_predictions = majority_voting(preds_test_model1, preds_test_model2, preds_test_model3)

            return jsonify({'predictions': final_predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(port=8000, debug=True)