import cv2
import numpy as np
import torch
from inference import predict
from class_map import classes
import os


def open_camera():
    candidates = [
        (0, cv2.CAP_DSHOW),
        (1, cv2.CAP_DSHOW),
        (0, cv2.CAP_MSMF),
        (0, cv2.CAP_ANY),
    ]

    for idx, backend in candidates:
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"Using camera index={idx}, backend={backend}")
                return cap
        cap.release()

    return None


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("outputs", exist_ok=True)

cap = open_camera()
if cap is None:
    print("❌ Cannot open camera")
    exit()

print("Press 's' to save, 'q' to quit")

# 🔥 防亂跳機制
last_pred_char = None
stable_pred_char = None
same_count = 0
required_same = 5   # 可以調整：越大越穩

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    # ROI
    x1, y1, x2, y2 = 200, 100, 400, 300
    roi = display[y1:y2, x1:x2]

    pred_char = None
    conf = 0.0
    canvas = np.zeros((28, 28), dtype=np.uint8)
    thresh_vis = np.zeros((200, 200), dtype=np.uint8)

    if roi.size > 0:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        thresh = cv2.medianBlur(thresh, 3)

        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        thresh_vis = cv2.resize(thresh, (200, 200), interpolation=cv2.INTER_NEAREST)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            filtered = []
            for c in contours:
                x_c, y_c, w_c, h_c = cv2.boundingRect(c)
                aspect_ratio = h_c / (w_c + 1e-5)

                if w_c > 25 and h_c > 25 and aspect_ratio < 4:
                    filtered.append(c)

            if filtered:
                largest = max(filtered, key=cv2.contourArea)
            else:
                largest = max(contours, key=cv2.contourArea)

            x, y, w, h = cv2.boundingRect(largest)

            if w > 25 and h > 25:
                pad = 12
                x = max(0, x - pad)
                y = max(0, y - pad)
                w = min(thresh.shape[1] - x, w + 2 * pad)
                h = min(thresh.shape[0] - y, h + 2 * pad)

                # square
                side = max(w, h)
                cx = x + w // 2
                cy = y + h // 2

                x = max(0, cx - side // 2)
                y = max(0, cy - side // 2)

                if x + side > thresh.shape[1]:
                    side = thresh.shape[1] - x
                if y + side > thresh.shape[0]:
                    side = thresh.shape[0] - y

                digit = thresh[y:y+side, x:x+side]

                if digit.size > 0:
                    digit = cv2.resize(
                        digit, (20, 20),
                        interpolation=cv2.INTER_AREA
                    )

                    canvas = np.zeros((28, 28), dtype=np.uint8)
                    canvas[4:24, 4:24] = digit

                    cv2.imwrite("outputs/debug_input.png", canvas)

                    img = canvas / 255.0
                    img = torch.tensor(
                        img, dtype=torch.float32
                    ).unsqueeze(0).unsqueeze(0).to(device)

                    # 🔥 提高 threshold
                    pred, conf = predict(img, threshold=0.6)

                    if pred is not None and pred < len(classes):
                        current_char = classes[pred]

                        if current_char == last_pred_char:
                            same_count += 1
                        else:
                            same_count = 1
                            last_pred_char = current_char

                        if same_count >= required_same:
                            stable_pred_char = current_char
                            pred_char = stable_pred_char
                        else:
                            pred_char = None

                        print("raw:", current_char, conf, "count:", same_count)
                    else:
                        pred_char = None
                        same_count = 0
                        last_pred_char = None

                cv2.rectangle(roi, (x, y), (x+side, y+side), (0, 255, 0), 2)

    # 顯示
    if pred_char:
        text = f"{pred_char} ({conf:.2f})"
    else:
        text = "..."

    cv2.putText(display, text, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 2)

    debug_vis = cv2.resize(canvas, (200, 200), interpolation=cv2.INTER_NEAREST)

    cv2.imshow("Realtime Recognition", display)
    cv2.imshow("Threshold", thresh_vis)
    cv2.imshow("Debug Input", debug_vis)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and pred_char:
        with open("outputs/result.txt", "a") as f:
            f.write(pred_char + "\n")

        cv2.imwrite(f"outputs/{pred_char}.png", canvas)
        print("Saved:", pred_char)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()