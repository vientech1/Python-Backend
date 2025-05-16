from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
from PIL import Image
import cv2
import numpy as np
import stone
import traceback
import base64
from face_shape_model import load_model, predict_face_shape


# Load the model once globally
face_shape_model = load_model("./model_85_nn_.pth")

app = Flask(__name__)

CORS(app)

@app.route('/', methods=['GET'])
def hello():
    return "<h1> App Working Fine</h1>"

@app.route('/api/result', methods=['POST'])
def process_image():
    try:
        data = request.get_json()
        if not data or 'image_data' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        image_data = base64.b64decode(data['image_data'])
        image_path = 'temp_image.jpg'
        with open(image_path, 'wb') as f:
            f.write(image_data)

        # 🧠 Step 1: Face shape prediction
        print("working")
        class_names = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
        result = predict_face_shape(image_path, face_shape_model)
        print("working2")

        face_shape_result = {
            "predicted": result["predicted"],
            "probabilities": result["probabilities"]
        }

        # 🎨 Step 2: Skin color analysis
        print("working3")
        skin_result = stone.process(image_path, image_type="color", return_report_image=True)
        serializable_skin = convert_numpy_to_serializable(skin_result)
        print("working4")

        os.remove(image_path)

        # 🔁 Combine both results
        return jsonify({
            "face_shape": face_shape_result,
            "skin_color": serializable_skin
        })

    except Exception as e:
        error_details = traceback.format_exc()
        return jsonify({'error': str(e), 'details': error_details}), 500



def convert_numpy_to_serializable(obj):
    try:
        if isinstance(obj, np.ndarray):
            if len(obj.shape) >= 2 and obj.shape[0] > 0 and obj.shape[1] > 0:
                try:
                    _, buffer = cv2.imencode('.jpg', obj)
                    img_str = base64.b64encode(buffer).decode('utf-8')
                    return f"data:image/jpeg;base64,{img_str}"
                except Exception as e:
                    return obj.tolist()
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.int, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                              np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        else:
            return obj
    except Exception as e:
        return str(obj)
    


if __name__ == '__main__':
    app.run(debug=True)
