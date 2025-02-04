import cv2
import numpy as np
import re
import os
from ultralytics import YOLO

class PlateRecognizer:
    def __init__(self, plate_model_path, char_model_path,
                 conf_threshold_plate=0.25, conf_threshold_char=0.25,
                 expansion_factor=0.15, row_threshold=20, 
                 min_width=32, min_height=16):
      
        self.plate_model = YOLO(plate_model_path)
        self.character_model = YOLO(char_model_path)
        self.char_mapping = self.get_character_mapping()
        
        self.conf_threshold_plate = conf_threshold_plate
        self.conf_threshold_char = conf_threshold_char
        self.expansion_factor = expansion_factor
        self.row_threshold = row_threshold
        self.min_width = min_width
        self.min_height = min_height

    @staticmethod
    def check_legit_plate(s):
        s_cleaned = re.sub(r'[.\-\s]', '', s)
        return 8 <= len(s_cleaned) <= 9

    @staticmethod
    def check_image_size(image, w_thres, h_thres):

        if w_thres is None:
            w_thres = 64
        if h_thres is None:
            h_thres = 64
        h, w, _ = image.shape
        return (w >= w_thres) and (h >= h_thres)

    @staticmethod
    def draw_text(img, text,
                  pos=(0, 0),
                  font=cv2.FONT_HERSHEY_SIMPLEX,
                  font_scale=1,
                  font_thickness=2,
                  text_color=(255, 255, 255),
                  text_color_bg=(0, 0, 0)):

        x, y = pos
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        bg_h = int(text_h * 1.05)
        cv2.rectangle(img, pos, (x + text_w, y + bg_h), text_color_bg, -1)
        pos = (x, y + text_h + font_scale)
        cv2.putText(img, text, pos, font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    @staticmethod
    def crop_expanded_plate(bbox, image, expansion_factor=0.15):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x_exp = int(w * expansion_factor)
        y_exp = int(h * expansion_factor)
        x1_exp = max(x1 - x_exp, 0)
        y1_exp = max(y1 - y_exp, 0)
        x2_exp = min(x2 + x_exp, image.shape[1])
        y2_exp = min(y2 + y_exp, image.shape[0])
        return image[y1_exp:y2_exp, x1_exp:x2_exp]

    @staticmethod
    def get_character_mapping():
        mapping = {i: str(i) for i in range(10)}
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for i, letter in enumerate(letters, start=10):
            mapping[i] = letter
        return mapping

    def extract_plate_characters(self, plate_img, conf_threshold=None):
        if conf_threshold is None:
            conf_threshold = self.conf_threshold_char

        results = self.character_model.predict(source=plate_img, conf=conf_threshold)
        result = results[0]
        
        if result.boxes is None or len(result.boxes) == 0:
            return "nan", result

        boxes = result.boxes.xyxy.cpu().numpy()  
        cls_indices = result.boxes.cls.cpu().numpy() 

        detections = []
        for box, cls in zip(boxes, cls_indices):
            x1 = int(box[0])
            y1 = int(box[1])
            char = self.char_mapping.get(int(cls), "")
            detections.append((x1, y1, char))
        
        detections.sort(key=lambda x: x[1])
        
        rows = []
        current_row = []
        current_y = None
        for det in detections:
            x, y, ch = det
            if current_y is None:
                current_y = y
                current_row.append(det)
            elif abs(y - current_y) <= self.row_threshold:
                current_row.append(det)
            else:
                rows.append(current_row)
                current_row = [det]
                current_y = y
        if current_row:
            rows.append(current_row)
        
        rows.sort(key=lambda row: row[0][1])
        
        ordered_chars = []
        for row in rows:
            row.sort(key=lambda x: x[0])
            for det in row:
                ordered_chars.append(det[2])
        
        plate_number = ''.join(ordered_chars)
        if self.check_legit_plate(plate_number):
            return plate_number, result
        else:
            return "nan", result

    def process_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print("Không đọc được ảnh:", image_path)
            return None
        output_image = image.copy()

        plate_results = self.plate_model.predict(source=image, conf=self.conf_threshold_plate, imgsz=320)
        plate_result = plate_results[0]

        if (plate_result.boxes is None) or (len(plate_result.boxes) == 0):
            print("Không phát hiện được biển số nào!")
            return output_image

        for detection in plate_result.boxes:
            bbox = detection.xyxy.cpu().numpy()[0].astype(int)
            x1, y1, x2, y2 = bbox
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            plate_crop = self.crop_expanded_plate(bbox, image, self.expansion_factor)
            if not self.check_image_size(plate_crop, self.min_width, self.min_height):
                continue

            plate_number, _ = self.extract_plate_characters(plate_crop, conf_threshold=self.conf_threshold_char)
            self.draw_text(output_image, plate_number, pos=(x1, y1 - 10),
                           text_color=(255, 255, 255), text_color_bg=(0, 0, 0))
        
        return output_image

def main():
    plate_model_path = "yolov8_plates.pt"
    char_model_path = "yolov8_characters.pt"
    recognizer = PlateRecognizer(plate_model_path, char_model_path)
    
    image_path = "a_164337.jpg" 
    output_image = recognizer.process_image(image_path)
    
    if output_image is not None:
        output_path = "output.jpg"
        cv2.imwrite(output_path, output_image)
        print(output_path)
        
        from google.colab.patches import cv2_imshow
        cv2_imshow(output_image)
    else:
        print("Lỗi")

if __name__ == "__main__":
    main()
