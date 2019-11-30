import cv2
import face_recognition
import os
import re


class Image:
    def __init__(self, image_path):
        self.image_path = image_path
        self.index = int(re.findall(r"\d+", self.image_path.split("/")[-1])[0])
        self.normalized_data = None
        self.valid = False

        self.data_directory_path = "data"
        if not os.path.exists(self.data_directory_path):
            os.mkdir(self.data_directory_path)

        self.image_directory_path = os.path.join(self.data_directory_path, "image")
        if not os.path.exists(self.image_path):
            os.mkdir(self.image_directory_path)

    def get_image_index(self):
        return self.index

    def get_image_path(self):
        return self.image_path

    def face_recognize(self):
        print("recognizing {}...".format(self.image_path))
        img = face_recognition.load_image_file(self.image_path)
        face_locations = face_recognition.face_locations(img)

        if len(face_locations) == 1:
            self.valid = True

            # crop
            cv_img = cv2.imread(self.image_path, 0)
            top, right, bottom, left = face_locations[0]
            cv_img = cv_img[top:bottom, left:right]

            # normalize
            cv_img = cv_img / 255.0
            cv_img = cv2.resize(cv_img, (48, 48))
            self.normalized_data = cv_img

            # save
            new_image_path = os.path.join(self.image_directory_path, self.image_path[11:].replace("/", "_"))
            cv2.imwrite(new_image_path, cv_img)

        return self.valid

    def get_normalized_data(self):
        return self.normalized_data

