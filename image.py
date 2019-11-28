class Image:
    def __init__(self, image_path, image_label):
        self.image_path = image_path
        if image_label == "Negative":
            self.image_label = -1
        elif image_label == "Neutral":
            self.image_label = 0
        else:
            self.image_label = 1

    def get_image_path(self):
        return self.image_path

    def get_label(self):
        return self.image_label