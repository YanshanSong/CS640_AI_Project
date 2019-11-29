import numpy as np
import csv
import os

from video import Video


class Preprocess:
    def __init__(self):
        self.images = []

        self.train_data_directory = os.path.join("data", "train")
        if not os.path.exists(self.train_data_directory):
            os.makedirs(self.train_data_directory)

    def get_all_images(self):
        with open("640ProjectData/Labels.csv", "r") as csvFile:
            reader = csv.reader(csvFile)
            reader = list(reader)
            for i in range(1, len(reader)):
                video_name = reader[i][0]
                video_label = reader[i][1]
                video = Video(video_name, video_label)
                self.images.extend(video.get_video_images())

    def get_train_data(self):
        X_train_path = os.path.join(self.train_data_directory, "X_train.npy")
        y_train_path = os.path.join(self.train_data_directory, "y_train.npy")

        if not os.path.exists(X_train_path):
            self.get_all_images()
            X_train_list = []
            y_train_list = []
            for image in self.images:
                if image.face_recognize():
                    X_train_list.append(image.get_normalized_data())
                    y_train_list.append(image.image_label)

            X_train = np.array(X_train_list)
            y_train = np.array(y_train_list).reshape(len(y_train_list), 1)
            np.save(X_train_path, X_train)
            np.save(y_train_path, y_train)
        else:
            X_train = np.load(X_train_path)
            y_train = np.load(y_train_path)

        return X_train, y_train


if __name__ == '__main__':
    preprocess = Preprocess()
    X_train, y_train = preprocess.get_train_data()
    print(X_train)
    print(y_train)