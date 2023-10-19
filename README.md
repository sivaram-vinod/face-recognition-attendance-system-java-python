# Face Recognition Attendance App

This application uses Android as the frontend and Python as the backend for face recognition-based attendance management. It involves the use of various technologies, including Python, Flask, TensorFlow, Android, and TensorFlow Lite.

## Usage

```bash
pip install virtualenv
python -m venv venv
source venv/bin/activate # venv/bin/activate.bat or venv/bin/activate.ps1
pip install -r requirements.txt
python generate.py
python train.py
python api.py # or python api.py -p 5001
```

or

```bash
pip3 install virtualenv
python3 -m venv venv
source venv/bin/activate # venv/bin/activate.bat or venv/bin/activate.ps1
pip3 install -r requirements.txt
python3 generate.py
python3 train.py
python api.py # or python api.py -p 5001
```

## Android App

1. Go to `FaceRecognition/app/src/main/java/org/tensorflow/lite/examples/detection/WebServices/ConstantString.java` and change the `URL` to point to your Python backend.

2. Adjust camera rotation settings if the bounding box for faces is not visible in the Android app. You can modify the `CameraActivity.java` file, especially at line 276.

3. Make sure to manage the asynchronous nature of face detection in the `DetectorActivity.java` file. When using face detection, you may need to handle the results asynchronously.

   - If the result is an empty list (i.e., `[]`), it means the face was not matched.

   - You might need to manage a flag or error handling for this case.
