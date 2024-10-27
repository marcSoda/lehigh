import os
from pathlib import Path
import argparse
import cv2
import numpy as np
import torch
import base64
import codecs
import pickle
# import torch.backends.cudnn as cudnn
from models.common import DetectPTBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import Annotator, colors, save_one_box

# Flask utils
from flask import Flask, request, jsonify

# Define a flask app
app = Flask(__name__)

model = None

@app.route('/predict', methods=["POST"])
def upload():
    if request.method == 'POST':
        # Get the data from post request
        data = request.form.get("data")
        if data != None:
            parser = argparse.ArgumentParser()
            data = detect(data)
            response = {
                        'status': 200,
                        'results': data,
                        }
            return jsonify(response)
        else:
            return "No data in body"

# ========= Configuration related to the Yolov5 ========≠
IMG_SIZE = (640, 640)
STRIDE = 32

def load_model(weights, device):
    model = torch.load(weights, map_location=device)
    return model['model'].float()

def detect(b64img):
    # =========== Load Image ===========
    npimg = np.fromstring(base64.b64decode(b64img), np.uint8)
    im0 = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    assert im0 is not None, f'Image Not Found'
    # =========== Preprocess Image ===========
    im = letterbox(im0, IMG_SIZE[0], stride=32, auto=True)[0] # Resize and padding
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im) # (3, 640, 640)
    im = torch.from_numpy(im).to(device).float().unsqueeze(0) # Convert image to tensor
    im /= 255  # 0 - 255 to 0.0 - 1.0
    ### =========== Inference ===========
    pred = model(im) # (1, 25200, 85) (center x, center y, width, height, conf, 80 class prob)
    ### =========== Post-processing ===========
    det = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, max_det=1000)[0]  # (N, 6)  (x1, y1, x2, y2, conf, cls)

    f = det.numpy().tolist()
    return f

if __name__ == '__main__':
    #device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device=torch.device("cpu")
    print("Inference on device: ", device)
    # =========== Load Model ===========
    print(f"Loading model")
    model = DetectPTBackend("./weights/yolov5s.pt", device=device)
    print(f"Loading model completed.")

    port = os.environ.get('PORT', 1701)
    app.run(debug=False, host='0.0.0.0', port=port)
