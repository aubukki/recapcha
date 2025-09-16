import pyautogui
from pynput.mouse import Controller, Button
import time
import cv2
from ultralytics import YOLO
import torch
import tkinter as tk
from PIL import Image, ImageTk
import win32api
import win32con
import math
import ctypes

#check if more than one object in box
def is_close_to_existing(new_point, existing_points, threshold=70):
    for (ex, ey) in existing_points:
        dist = math.hypot(new_point[0] - ex, new_point[1] - ey)
        if dist < threshold:
            return True
    return False

# Load user32.dll
user32 = ctypes.windll.user32

# Get screen resolution
screen_w = user32.GetSystemMetrics(0)
screen_h = user32.GetSystemMetrics(1)
mouse = Controller()

# Load YOLO12 model
model_path = "yolo12x.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(model_path)
model.to(device)

# Load your capcha image
input_image_path = "output.png"
img = cv2.imread(input_image_path)
if img is None:
    raise FileNotFoundError(f"Image not found: {input_image_path}")

annotated_img = img.copy()

# Target labels and confidence threshold
target_labels = ["traffic light"]
conf_threshold = 0.5

# Run YOLO detection
results = model(img, device=device)

# Extract detected target boxes
detected_boxes = []

img_h, img_w, _ = img.shape
screen_w, screen_h = pyautogui.size()
last_detection = None   # store last detection
skip_once = False       # flag to allow skipping for 1 cycle

for i in range(1):
 for r in results:
    boxes = r.boxes.xyxy.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()

    for box, cls, conf in zip(boxes, classes, confs):
        label_name = model.names[int(cls)]

        if label_name.lower() in target_labels and conf > conf_threshold:
            x1, y1, x2, y2 = map(int, box)
            label = f"{label_name}: {conf:.2f}"
            w = x2 - x1
            h = y2 - y1
# if the object is big the refion gets selectet
            if w >= 100 and h >= 100:
                 step = 120 # pixels between points
                 print("gaY")
                 detected_boxes.extend(
                [(px, py) 
         for px in range(x1, x2 + 0, step) 
         for py in range(y1, y2 +0, step)])
            else:
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_img, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                print("aaa")
            # Map to screen coordinates
                screen_x = center_x
                screen_y = center_y
                new_detection = (screen_x, screen_y)
                if not is_close_to_existing(new_detection, detected_boxes, threshold=70):
                       detected_boxes.append(new_detection)
                else:
                        print("Skipped duplicate detection")
                        cv2.imwrite("annotated_output.png", annotated_img)


# Save annotated image
cv2.imwrite("annotated_output.png", annotated_img)

# ------------------ Tkinter Display ------------------
def show_image_and_move_mouse():
    root = tk.Tk()
    root.title("Annotated Image")

    # Remove window decorations and make small
    root.overrideredirect(True)  # Removes title bar
    root.geometry(f"{img_w}x{img_h}+0+0")  # Position at (0,0)

    img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)

    label = tk.Label(root, image=img_tk)
    label.pack()

    # Update window so the image appears
    root.update()

    # Move mouse to detected objects automatically
    for (screen_x, screen_y) in detected_boxes:
        sigma_boy = (screen_x, screen_y)  # detected in screenshot
        print(sigma_boy)  # Output: (960, 466)
        sigma, boy = sigma_boy
        abs_x = int(sigma * 65535 / screen_w)
        abs_y = int(boy * 65535 / screen_h)

# Flags for absolute move
        MOUSEEVENTF_MOVE = 0x0001
        MOUSEEVENTF_ABSOLUTE = 0x8000

# Move the mouse
        user32.mouse_event(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, abs_x, abs_y, 0, 0)
        time.sleep(0.8)
        time.sleep(0.3)

    root.mainloop()

show_image_and_move_mouse()
