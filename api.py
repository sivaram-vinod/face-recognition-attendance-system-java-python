from flask import Flask, jsonify, request
import pickle
import face_recognition
import time
from io import BytesIO
import base64
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "-p",
    "--port",
    type=int,
    default=4040,
    help="specify the port number, default is 4040.",
)

args = parser.parse_args()

app = Flask(__name__)

app.config["ALLOWED_EXTENSIONS"] = set(["jpg", "jpeg", "png"])


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


@app.route("/api/facedetect", methods=["GET", "POST"])
def upload_image():
    if request.method == "GET":
        resp = jsonify({"message": "Error", "data'": "Method not allowed"})
        resp.status_code = 405
        return resp
    else:
        requestData = request.form
        if "face" not in requestData:
            resp = jsonify({"message": "Error", "data'": "No face found"})
            resp.status_code = 500
            return resp

        file = BytesIO(base64.decodebytes(requestData["face"].encode()))
        start = time.time()
        results = []
        with open("trained_knn_model.clf", "rb") as f:
            knn_clf = pickle.load(f)

            image = face_recognition.load_image_file(file)
            X_face_locations = face_recognition.face_locations(image)

            if len(X_face_locations) != 0:
                faces_encodings = face_recognition.face_encodings(
                    image, known_face_locations=X_face_locations
                )

                closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
                are_matches = [
                    closest_distances[0][i][0] <= 0.4
                    for i in range(len(X_face_locations))
                ]
                predictions = [
                    (pred, loc) if rec else ("unknown", loc)
                    for pred, loc, rec in zip(
                        knn_clf.predict(faces_encodings), X_face_locations, are_matches
                    )
                ]
                lp = 0
                for name, (top, right, bottom, left) in predictions:
                    resarray = {}
                    resarray["name"] = name
                    resarray["accuracy"] = closest_distances[0][lp][0]
                    results.append(resarray)
                    lp = lp + 1

        print(results)
        if results is None:
            resp = jsonify({"message": "Error", "data'": "No face found"})
            resp.status_code = 500
            return resp
        else:
            resp = jsonify({"message": "success", "data": results})
            resp.status_code = 200
            return resp


if __name__ == "__main__":
    app.run(port=args.port, host="0.0.0.0")
