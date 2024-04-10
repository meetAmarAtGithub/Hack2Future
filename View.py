# views.py

import cv2
import numpy as np
from django.shortcuts import render
from .forms import UploadImageForm
from .yolo_model import YOLOModel  # Import your YOLO model

def detect_objects(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            image_file = request.FILES['image']
            img = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
            
            # Perform object detection using YOLO model
            yolo_model = YOLOModel()
            detected_objects = yolo_model.detect_objects(img)

            return render(request, 'results.html', {'image_file': image_file, 'detected_objects': detected_objects})

    else:
        form = UploadImageForm()

    return render(request, 'upload.html', {'form': form})
