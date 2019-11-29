import cv2
#import face_recognition
import os


class Image:
    def __init__(self, image_path, image_label=None):
        self.image_path = image_path
        self.normalized_data = None

        if image_label:
            if image_label == "Negative":
                self.image_label = -1
            elif image_label == "Neutral":
                self.image_label = 0
            else:
                self.image_label = 1
        else:
            self.image_label = -2

        self.valid = False

    def get_image_path(self):
        return self.image_path

    def get_image_label(self):
        return self.image_label

    # def face_recognize(self):
    #     print("recognizing {}...".format(self.image_path))
    #     img = face_recognition.load_image_file(self.image_path)
    #     face_locations = face_recognition.face_locations(img)
    #
    #     if len(face_locations) == 1:
    #         self.valid = True
    #
    #         # crop
    #         cv_img = cv2.imread(self.image_path, 0)
    #         top, right, bottom, left = face_locations[0]
    #         cv_img = cv_img[top:bottom, left:right]
    #
    #         # normalize
    #         cv_img = cv_img / 255.0
    #         cv_img = cv2.resize(cv_img, (48, 48))
    #         self.normalized_data = cv_img
    #
    #         # save
    #         # self.image_path = os.path.join("data/image", self.image_path[10:].replace("/", "_"))
    #         # cv2.imwrite(self.image_path, cv_img)
    #
    #     return self.valid

    def get_normalized_data(self):
        return self.normalized_data


if __name__ == '__main__':
    image = Image("repository/amy.4xD6Ec4g5N0.00.v.mp4/image_1.jpg")
    image.face_recognize()

