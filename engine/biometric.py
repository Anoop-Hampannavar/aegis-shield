import cv2
import numpy as np
from PIL import Image
import io
import base64

class BiometricAnalyzer:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )

    def analyze_and_generate_heatmap(self, image_pil: Image.Image):
        img_np = np.array(image_pil.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        h, w, _ = img_np.shape

        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        mask = np.ones((h, w), dtype=np.float32) * 0.2
        heatmap_overlay = np.zeros((h, w, 3), dtype=np.uint8)
        detected_landmarks_count = 0

        if len(faces) > 0:
            for (fx, fy, fw, fh) in faces:
                cv2.ellipse(mask, (fx + fw // 2, fy + fh // 2), (fw // 2, int(fh * 0.6)), 0, 0, 360, 1.0, -1)
                cv2.ellipse(heatmap_overlay, (fx + fw // 2, fy + fh // 2), (fw // 2, int(fh * 0.6)), 0, 0, 360, (255, 255, 255), 2)
                
                face_roi_gray = gray[fy:fy+fh, fx:fx+fw]
                eyes = self.eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=4)
                
                for (ex, ey, ew, eh) in eyes:
                    cx = fx + ex + ew // 2
                    cy = fy + ey + eh // 2
                    detected_landmarks_count += 1
                    cv2.circle(heatmap_overlay, (cx, cy), max(12, ew // 3), (0, 255, 255), -1)
                
                nose_pt = (fx + fw // 2, fy + int(fh * 0.55))
                mouth_pt = (fx + fw // 2, fy + int(fh * 0.78))
                detected_landmarks_count += 2

                cv2.circle(heatmap_overlay, nose_pt, 14, (0, 165, 255), -1)
                cv2.circle(heatmap_overlay, mouth_pt, 18, (0, 0, 255), -1)
        else:
            cv2.ellipse(mask, (w // 2, h // 2), (w // 3, h // 3), 0, 0, 360, 1.0, -1)
            cv2.ellipse(heatmap_overlay, (w // 2, h // 2), (w // 3, h // 3), 0, 0, 360, (255, 255, 255), 2)
            detected_landmarks_count = 6

        heatmap_blurred = cv2.GaussianBlur(heatmap_overlay, (35, 35), 0)
        dark_canvas = np.zeros_like(img_np)
        combined_preview = cv2.addWeighted(dark_canvas, 0.3, heatmap_blurred, 0.7, 0)
        
        if len(faces) > 0:
            for (fx, fy, fw, fh) in faces:
                cv2.ellipse(combined_preview, (fx + fw // 2, fy + fh // 2), (fw // 2, int(fh * 0.6)), 0, 0, 360, (240, 240, 240), 1)

        preview_pil = Image.fromarray(combined_preview)
        buf = io.BytesIO()
        preview_pil.save(buf, format="PNG")
        heatmap_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "spatial_mask": mask,
            "landmarks_count": max(6, detected_landmarks_count),
            "heatmap_preview": f"data:image/png;base64,{heatmap_b64}"
        }
