from ultralytics import YOLO
import cv2
import sys
import os


def load_model(model_size="n"):
    """
    Model load karta hai. Sizes:
    n = nano (sabse fast, halka)
    s = small
    m = medium
    l = large
    x = extra large (sabse accurate, but slow)
    """
    model_name = f"yolov8{model_size}.pt"
    print(f"[INFO] Loading model: {model_name} ...")
    model = YOLO(model_name)  # pehli baar run karne par automatic download hoga
    print("[INFO] Model loaded successfully!")
    return model


def detect_on_image(model, image_path, save_path="output_image.jpg"):
    """Single image par object detection karo aur result save karo"""
    if not os.path.exists(image_path):
        print(f"[ERROR] Image nahi mili: {image_path}")
        return

    results = model(image_path)  # detection chalao

    for result in results:
        boxes = result.boxes  # detected boxes
        print(f"[INFO] Total objects detected: {len(boxes)}")

        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            print(f"  -> {class_name} | Confidence: {confidence:.2f} | "
                  f"Box: ({int(x1)}, {int(y1)}), ({int(x2)}, {int(y2)})")

        # annotated image (boxes + labels) save karo
        annotated_frame = result.plot()
        cv2.imwrite(save_path, annotated_frame)
        print(f"[INFO] Result saved at: {save_path}")


def detect_on_video(model, video_path, save_path="output_video.mp4"):
    """Video file par object detection karo (frame by frame)"""
    if not os.path.exists(video_path):
        print(f"[ERROR] Video nahi mili: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    print("[INFO] Processing video... (Ctrl+C se rok sakte ho)")
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"[INFO] {frame_count} frames processed...")

    cap.release()
    out.release()
    print(f"[INFO] Done! Output saved at: {save_path}")


def detect_on_webcam(model):
    """Live webcam se real-time object detection"""
    cap = cv2.VideoCapture(0)  # 0 = default webcam

    if not cap.isOpened():
        print("[ERROR] Webcam access nahi ho paayi.")
        return

    print("[INFO] Webcam start ho gaya. Band karne ke liye 'q' dabao.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()

        cv2.imshow("Object Detection - Press 'q' to quit", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    # --------- YAHAN SE SETTINGS BADLO ---------
    MODEL_SIZE = "n"          # n/s/m/l/x (n = fastest, recommended for start)
    MODE = "webcam"            # "image", "video", ya "webcam"
    INPUT_PATH = "pj.jpg"   # apni image/video ka path yahan daalo
    # --------------------------------------------

    model = load_model(MODEL_SIZE)

    if MODE == "image":
        detect_on_image(model, INPUT_PATH)
    elif MODE == "video":
        detect_on_video(model, INPUT_PATH)
    elif MODE == "webcam":
        detect_on_webcam(model)
    else:
        print("[ERROR] MODE ko 'image', 'video' ya 'webcam' set karo")



# Camera Access karana ka liya
# if __name__ == "__main__":
#     MODEL_SIZE = "n"
#     MODE = "webcam"

#     model = load_model(MODEL_SIZE)

#     if MODE == "webcam":
#         detect_on_webcam(model)

#     else:
#         print("[ERROR] MODE ko 'webcam' set karo")