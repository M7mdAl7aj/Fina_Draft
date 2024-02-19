import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os
from utilities import get_path, show_image, show_image_with_matplotlib
import transformers

class ObjectDetector:
    def __init__(self):
        self.model = None
        self.processor = None
        self.model_name = None

    def load_model(self, model_name='detic', pretrained=True, model_version='yolov5s'):
        """
        Load the specified object detection model.
        :param model_name: Name of the model to load.
        :param pretrained: Boolean indicating if pretrained model should be used.
        :param model_version: Version of the model, applicable for YOLOv5.
        """
        self.model_name = model_name
        if model_name == 'detic':
            self.load_detic_model(pretrained)
        elif model_name == 'yolov5':
            self.load_yolov5_model(pretrained, model_version)
        else:
            raise ValueError("Unsupported model name")


    def load_detic_model(self, pretrained):
        """Load the Detic model."""
        try:
            model_path = get_path('deformable-detr-detic', 'Models')
            print(model_path)
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
            self.processor = AutoImageProcessor.from_pretrained(model_path)
            self.model = AutoModelForObjectDetection.from_pretrained(model_path)
        except Exception as e:
            print(f"Error loading Detic model: {e}")


    def load_yolov5_model(self, pretrained, model_version):
        """Load the YOLOv5 model."""
        try:
            model_path = get_path('yolov5', 'Models')
            if model_path and os.path.exists(model_path):
                with os.scandir(model_path) as main_dir:
                    self.model = torch.hub.load(model_path, model_version, pretrained=pretrained, source="local")
            else:
                self.model = torch.hub.load('ultralytics/yolov5', model_version, pretrained=pretrained)
        except Exception as e:
            print(f"Error loading YOLOv5 model: {e}")


    def process_image(self, image_path: str) -> Image.Image:
        """
        Process the image from the given path.
        :param image_path: Path to the image file.
        :return: Processed image.
        """
        with Image.open(image_path) as image:
            return image.convert("RGB")


    def detect_objects(self, image: Image.Image, threshold: float = 0.4):
        """
        Detect objects in the given image.
        :param image: Image in which to detect objects.
        :param threshold: Detection threshold.
        :return: Tuple of detected objects string and list.
        """
        detected_objects_str, detected_objects_list = "", []
        if self.model_name == 'detic':
            detected_objects_str, detected_objects_list = self.detect_with_detic(image, threshold)
        elif self.model_name == 'yolov5':
            detected_objects_str, detected_objects_list = self.detect_with_yolov5(image, threshold)
        return detected_objects_str.strip(), detected_objects_list


    def detect_with_detic(self, image: Image.Image, threshold: float):
        """Detect objects using Detic model."""
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[
            0]

        detected_objects_str = ""
        detected_objects_list = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if score >= threshold:
                label_name = self.model.config.id2label[label.item()]
                box_rounded = [round(coord, 2) for coord in box.tolist()]
                certainty = round(score.item() * 100, 2)
                detected_objects_str += f"{{object: {label_name}, bounding box: {box_rounded}, certainty: {certainty}%}}\n"
                detected_objects_list.append((label_name, box_rounded, certainty))
        return detected_objects_str, detected_objects_list


    def detect_with_yolov5(self, image: Image.Image, threshold: float):
        """Detect objects using YOLOv5 model."""

        cv2_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        results = self.model(cv2_img)

        detected_objects_str = ""
        detected_objects_list = []
        for *bbox, conf, cls in results.xyxy[0]:
            if conf >= threshold:
                label_name = results.names[int(cls)]
                box_rounded = [round(coord.item(), 2) for coord in bbox]  # Convert each tensor to float and round
                certainty = round(conf.item() * 100, 2)
                detected_objects_str += f"{{object: {label_name}, bounding box: {box_rounded}, certainty: {certainty}%}}\n"
                detected_objects_list.append((label_name, box_rounded, certainty))
        return detected_objects_str, detected_objects_list


    def draw_boxes(self, image: Image.Image, detected_objects: list, show_confidence: bool = True) -> Image.Image:
        """
        Draw bounding boxes around detected objects in the image.
        :param image: Image on which to draw.
        :param detected_objects: List of detected objects.
        :param show_confidence: Boolean to show confidence scores.
        :return: Image with drawn boxes.
        """
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()

        colors = ["red", "green", "blue", "yellow", "purple", "orange"]
        label_color_map = {}

        for label_name, box, score in detected_objects:
            if label_name not in label_color_map:
                label_color_map[label_name] = colors[len(label_color_map) % len(colors)]

            color = label_color_map[label_name]
            draw.rectangle(box, outline=color, width=3)

            label_text = f"{label_name}"
            if show_confidence:
                label_text += f" ({round(score, 2)}%)"
            draw.text((box[0], box[1]), label_text, fill=color, font=font)

        return image


if __name__=="__main__":


    detector = ObjectDetector()
    image_path = get_path('horse.jpg', 'Sample_Images')
    print(image_path)

    detector.load_model('yolov5')  # pass either 'detic' or 'yolov5'

    image = detector.process_image(image_path)
    detected_objects_string, detected_objects_list = detector.detect_objects(image, threshold=0.2)
    image_with_boxes = detector.draw_boxes(image, detected_objects_list, show_confidence=False)
    print(detected_objects_string)
    show_image(image_with_boxes)
    #show_image_with_matplotlib(image_path)


