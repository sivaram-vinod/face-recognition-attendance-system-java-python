<img src="https://1.bp.blogspot.com/-N-XwxleEyOo/WYQEtqUZGnI/AAAAAAAAwRI/Klh5vIblR_EzyXjHsm1zh5WP3hWZMaciACLcBGAs/s1600/SRM%2BLogo.png" height=70>

# Face Recognition Attendance App

This application uses Android as the frontend and Python as the backend for face recognition-based attendance management. It involves the use of various technologies, including Python, Flask, TensorFlow, Android, and TensorFlow Lite.

We created this app together to say no to proxy.

## Usage

The default port for api url is 4040, you can use custom port number by passing an argument `python api.py -p 5023`.

```bash
pip install virtualenv
python -m venv venv
venv/Scripts/activate.bat # source venv/bin/activate or venv/Scripts/activate.ps1
pip install -r requirements.txt
python generate.py
python train.py
python api.py # or python api.py -p 5001
```

or

```bash
pip3 install virtualenv
python3 -m venv venv
venv/Scripts/activate.bat # source venv/bin/activate or venv/Scripts/activate.ps1
pip3 install -r requirements.txt
python3 generate.py
python3 train.py
python api.py # or python api.py -p 5001
```

## Credits

- [Flask](https://flask.palletsprojects.com/en/3.0.x/)
- [Tensorflow Lite](https://www.tensorflow.org/lite)
- [OpenCV](https://opencv.org)

## Authors

- [**Manish Kumar**](https://github.com/its-manishks)
- [**Sivaram Vinod**](https://github.com/sivaram-vinod)