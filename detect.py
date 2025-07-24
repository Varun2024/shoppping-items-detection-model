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


model = YOLO("runs/detect/train6/weights/best.pt")  


cap = cv2.VideoCapture(0)
reader = easyocr.Reader(['en'], gpu=False)  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    results = model(frame)

    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            
            
            cropped = frame[y1:y2, x1:x2]
            
            
            ocr_results = reader.readtext(cropped)
            text = ocr_results[0][1] if ocr_results else "Unknown"

            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            print(text)

    
    cv2.imshow("Shopping Item Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
