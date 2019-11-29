import cv2
import os
import math
from image import Image


class Video:
    def __init__(self, video_name, video_label=None):
        # if not exists, create "repository" repository
        self.image_repository_path = "repository"
        if not os.path.exists(self.image_repository_path):
            os.mkdir(self.image_repository_path)

        self.video_name = video_name
        self.video_label = video_label
        self.video_path = os.path.join("640ProjectData/presidential_videos", self.video_name)
        self.videoCapture = cv2.VideoCapture(self.video_path)
        self.video_images = []

    def extract_images_from_video(self):
        # if not exists,
        # create a directory in the "repository" directory to store images extracted from the current video
        image_directory_path = os.path.join(self.image_repository_path, self.video_name)
        if not os.path.exists(image_directory_path):
            os.mkdir(image_directory_path)
            print("Processing {}...".format(self.video_name))

            fps = self.get_fps()

            # save images
            image_count = 0
            frame_count = 0
            success, frame = self.videoCapture.read()
            while success:
                frame_count += 1
                if frame_count % math.ceil(2 * fps) == 0:
                    image_count += 1
                    image_path = os.path.join(image_directory_path, "image_{}.jpg".format(image_count))
                    cv2.imwrite(image_path, frame)
                    image = Image(image_path, self.video_label)
                    self.video_images.append(image)
                success, frame = self.videoCapture.read()
        else:
            for root, dirs, files in os.walk(image_directory_path):
                for file in files:
                    image_path = os.path.join(image_directory_path, file)
                    image = Image(image_path, self.video_label)
                    self.video_images.append(image)

    def get_fps(self):
        return self.videoCapture.get(5)

    def get_video_images(self):
        self.extract_images_from_video()
        return self.video_images



