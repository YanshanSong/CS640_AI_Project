import numpy as np
import csv
import os

from video import Video


class Preprocess:
    def __init__(self):
        self.videos = []

        self.data_directory = "data"
        if not os.path.exists(self.data_directory):
            os.mkdir(self.data_directory)

        self.data_image_directory = os.path.join(self.data_directory, "image")
        if not os.path.exists(self.data_image_directory):
            os.mkdir(self.data_image_directory)

    def get_all_videos(self):
        with open("640ProjectData/Labels.csv", "r") as csvFile:
            reader = csv.reader(csvFile)
            reader = list(reader)
            for i in range(1, len(reader)):
                video_path = os.path.join("640ProjectData/presidential_videos", reader[i][0])
                video_label = reader[i][1]
                video = Video(video_path, video_label)
                self.videos.append(video)
        return self.videos

    def get_all_data(self):
        X_path = os.path.join(self.data_directory, "X.npy")
        y_path = os.path.join(self.data_directory, "y.npy")

        if not os.path.exists(X_path):
            self.get_all_videos()
            X_list = []
            y_list = []

            for video in self.videos:
                if video.generate_normalized_data():
                    X_list.append(video.get_normalized_data())
                    y_list.append(video.get_video_label())

            X = np.array(X_list)
            y = np.array(y_list).reshape(len(y_list), 1)
            np.save(X_path, X)
            np.save(y_path, y)
        else:
            X = np.load(X_path)
            y = np.load(y_path)

        return X, y


if __name__ == '__main__':
    preprocess = Preprocess()
    X, y = preprocess.get_all_data()
    print(X.shape)
    print(y.shape)
