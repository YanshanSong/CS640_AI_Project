import cv2
import os
import math
from image import Image
import numpy as np


class Video:
    def __init__(self, video_path, video_label=None):
        # if not exists, create "repository" repository
        self.repository_path = "repository"
        if not os.path.exists(self.repository_path):
            os.mkdir(self.repository_path)

        self.video_path = video_path
        self.video_name = self.video_path.split("/")[-1]
        if video_label:
            if video_label == "Negative":
                self.video_label = 0
            elif video_label == "Neutral":
                self.video_label = 1
            else:
                self.video_label = 2
        else:
            self.video_label = -1

        self.valid = False
        self.video_images = []
        self.normalized_data = None

    def extract_images_from_video(self):
        # if not exists,
        # create a directory in the "repository" directory to store images extracted from the current video
        directory_path = os.path.join(self.repository_path, self.video_name)
        if not os.path.exists(directory_path):
            os.mkdir(directory_path)
            print("Processing {}...".format(self.video_name))
            # save images
            image_count = 0
            frame_count = 0
            video_capture = cv2.VideoCapture(self.video_path)
            # get fps
            fps = video_capture.get(5)
            success, frame = video_capture.read()
            while success:
                frame_count += 1
                if frame_count % math.ceil(fps / 2) == 0:
                    image_count += 1
                    image_path = os.path.join(directory_path, "image_{}.jpg".format(image_count))
                    cv2.imwrite(image_path, frame)
                    image = Image(image_path)
                    self.video_images.append(image)
                success, frame = video_capture.read()
        else:
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    image_path = os.path.join(directory_path, file)
                    image = Image(image_path)
                    self.video_images.append(image)

            self.video_images.sort(key=lambda i: i.get_image_index())

    def generate_normalized_data(self):
        self.extract_images_from_video()
        data_list = []
        for video_image in self.video_images:
            if video_image.face_recognize():
                data_list.append(video_image.get_normalized_data())
            if len(data_list) == 10:
                self.valid = True
                self.normalized_data = np.array(data_list)
                break

        return self.valid

    def get_normalized_data(self):
        return self.normalized_data

    def get_video_label(self):
        return self.video_label


if __name__ == '__main__':
    video = Video("640ProjectData/presidential_videos/donald.9Ct43xvDYDE.00.mp4")
    rs = video.generate_normalized_data();
    print(rs)

