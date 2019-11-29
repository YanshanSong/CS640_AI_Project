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
        trainX_path = os.path.join(self.train_data_directory, "trainX.npy")
        trainY_path = os.path.join(self.train_data_directory, "trainY.npy")

        if not os.path.exists(trainX_path):
            self.get_all_images()
            trainX_list = []
            trainY_list = []
            for image in self.images:
                if image.face_recognize():
                    trainX_list.append(image.get_normalized_data())
                    trainY_list.append(image.image_label)

            trainX = np.array(trainX_list)
            trainY = np.array(trainY_list).reshape(len(trainY_list), 1)
            np.save(trainX_path, trainX)
            np.save(trainY_path, trainY)
        else:
            trainX = np.load(trainX_path)
            trainY = np.load(trainY_path)

        return trainX, trainY


if __name__ == '__main__':
    preprocess = Preprocess()
    trainX, trainY = preprocess.get_train_data()



