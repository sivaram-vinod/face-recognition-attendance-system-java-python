from sklearn import neighbors
import os
import os.path
import pickle
import face_recognition
import numpy as np

if os.path.exists("students/.DS_Store"):
    os.remove("students/.DS_Store")


def train(
    train_dir,
    model_save_path=None,
    n_neighbors=None,
    knn_algo="ball_tree",
    verbose=False,
):
    encodings = []
    names = []

    train_dir = os.listdir("students/")
    print(train_dir)

    for person in train_dir:
        pix = os.listdir("students/" + person)

        for person_img in pix:
            print("students/" + person + "/" + person_img)
            face = face_recognition.load_image_file(
                "students/" + person + "/" + person_img
            )
            print(face.shape)

            height, width, _ = face.shape
            face_location = (0, width, height, 0)
            print(width, height)

            face_enc = face_recognition.face_encodings(
                face, known_face_locations=[face_location]
            )

            face_enc = np.array(face_enc)
            face_enc = face_enc.flatten()

            encodings.append(face_enc)
            names.append(person)

    print(np.array(encodings).shape)
    print(np.array(names).shape)

    knn_clf = neighbors.KNeighborsClassifier(
        n_neighbors=n_neighbors, algorithm=knn_algo, weights="distance"
    )
    knn_clf.fit(encodings, names)

    if model_save_path is not None:
        with open(model_save_path, "wb") as f:
            pickle.dump(knn_clf, f)

    return knn_clf


if __name__ == "__main__":
    print("Training KNN classifier...")
    classifier = train(
        "students", model_save_path="trained_knn_model.clf", n_neighbors=2
    )
    print("\nTraining complete!")
    print("Trained model saved to \"trained_knn_model.clf\"")
    print(f"Trained {len(classifier.classes_)} students in the Attendance System.\n")

    choice = input(str("Do you want to start the server? (y/n): "))
    if choice == "y":
        os.system("python api.py")
    else:
        print("Exiting...")
        exit(0)
