# for detection of shopping items

# from ultralytics import YOLO

# model = YOLO('runs/detect/train6/weights/best.pt') 
# from ultralytics import YOLO


# model.predict(source=0, show=True, conf=0.25) 



# for detection of item along with their labels(temp fix)
import cv2
import torch
from ultralytics import YOLO
import easyocr

# Load your trained YOLO model
model = YOLO("runs/detect/train6/weights/best.pt")  # update path as needed

# Start webcam
cap = cv2.VideoCapture(0)
reader = easyocr.Reader(['en'], gpu=False)  # GPU=True if available

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection
    results = model(frame)

    # Draw results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            
            # Crop detected item
            cropped = frame[y1:y2, x1:x2]
            
            # OCR on the cropped item
            ocr_results = reader.readtext(cropped)
            text = ocr_results[0][1] if ocr_results else "Unknown"

            # Draw box and text
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            print(text)

    # Show the frame
    cv2.imshow("Shopping Item Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
