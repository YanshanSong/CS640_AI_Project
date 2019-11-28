from video import Video
import csv


class Main:
    def __init__(self):
        self.images = []

    def preprocess(self):
        with open("640ProjectData/Labels.csv", "r") as csvFile:
            reader = csv.reader(csvFile)
            reader = list(reader)
            for i in range(1, len(reader)):
                video_name = reader[i][0]
                video_label = reader[i][1]
                video = Video(video_name, video_label)
                self.images.extend(video.get_video_images())

    def get_all_images(self):
        self.preprocess()
        return self.images


if __name__ == '__main__':
    main = Main()
    images = main.get_all_images()
    for image in images:
        print(image.get_image_path(), image.get_label())



