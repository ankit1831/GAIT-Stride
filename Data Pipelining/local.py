import cv2
import os
import shutil
import numpy as np
import time
import threading
import requests
from ultralytics import YOLO
from flask import Flask, jsonify
from flask_cors import CORS

# ==========================================
# GLOBAL VARIABLES
# ==========================================
latest_prediction = {"text": "", "timestamp": 0}
camera_active = False  # <--- THE MASTER SWITCH

# ==========================================
# 0. MICRO-SERVER (THE "EAR" FOR THE WEB APP)
# ==========================================
app = Flask(__name__)
CORS(app) 

# --- ADD THIS NEW BLOCK ---
@app.route('/', methods=['GET'])
def home_dashboard():
    # This creates a simple web page with two buttons so you can test it easily!
    return '''
    <html>
        <head><title>Gait Bridge Control</title></head>
        <body style="text-align: center; padding-top: 50px; font-family: Arial, sans-serif; background-color: #1e1e1e; color: white;">
            <h2>Local Camera Bridge is Active</h2>
            <p>Click below to test the hardware.</p>
            <br>
            <button onclick="fetch('/start_camera')" style="padding: 15px 30px; font-size: 18px; background-color: #28a745; color: white; border: none; border-radius: 5px; cursor: pointer; margin: 10px;">Start Camera</button>
            <button onclick="fetch('/stop_camera')" style="padding: 15px 30px; font-size: 18px; background-color: #dc3545; color: white; border: none; border-radius: 5px; cursor: pointer; margin: 10px;">Stop Camera</button>
        </body>
    </html>
    '''
# --------------------------

@app.route('/start_camera', methods=['GET', 'POST'])
def start_cam():
    global camera_active
    if not camera_active:
        camera_active = True
        return jsonify({"status": "success", "message": "Camera spinning up..."}), 200
    return jsonify({"status": "ignored", "message": "Camera is already running!"}), 200

@app.route('/stop_camera', methods=['GET', 'POST'])
def stop_cam():
    global camera_active
    camera_active = False
    return jsonify({"status": "success", "message": "Camera shutting down..."}), 200

def run_flask_server():
    # Runs quietly in the background on port 5001
    app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)

# ==========================================
# 1. THE YOLO SILHOUETTE EXTRACTOR (Unchanged)
# ==========================================
def process_video_to_silhouettes_custom(
    video_path="temp_gait.avi", 
    output_folder="live_test_folder", 
    img_size=(64, 64),
    inference_size=1080,        
    confidence_threshold=0.8,   
    extract_every_n_frame=3     
):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    
    print("Loading YOLOv8 Segmentation Model...")
    model = YOLO("yolov8n-seg.pt")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return
        
    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        frame_count += 1
        if frame_count % extract_every_n_frame != 0: continue
        
        results = model.predict(frame, classes=[0], retina_masks=True, imgsz=inference_size, conf=confidence_threshold, verbose=False)
        binary_silhouette = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        
        for r in results:
            if r.masks is not None:
                mask_data = r.masks.data[0].cpu().numpy()
                binary_silhouette = np.maximum(binary_silhouette, (mask_data > 0.5).astype(np.uint8) * 255)

        if np.sum(binary_silhouette) > 0:
            x, y, w, h = cv2.boundingRect(binary_silhouette)
            pad_w = int(w * 0.10); pad_h = int(h * 0.10)
            x1 = max(x - pad_w, 0); y1 = max(y - pad_h, 0)
            x2 = min(x + w + pad_w, frame.shape[1]); y2 = min(y + h + pad_h, frame.shape[0])
            
            cropped_person = binary_silhouette[y1:y2, x1:x2]
            final_silhouette = cv2.resize(cropped_person, img_size)
            
            save_path = os.path.join(output_folder, f"frame_{saved_count:04d}.png")
            cv2.imwrite(save_path, final_silhouette)
            saved_count += 1

    cap.release()
    print(f"\n[SUCCESS] Extracted {saved_count} silhouettes!")

# ==========================================
# 2. THE NETWORK & CLEANUP FUNCTION (Unchanged)
# ==========================================
def send_to_endpoint_and_cleanup(folder_path, video_path, endpoint_url="http://140.245.221.62:8000/predict"):
    global latest_prediction
    zip_filename = f"{folder_path}.zip"
    shutil.make_archive(folder_path, 'zip', folder_path)
    
    try:
        with open(zip_filename, 'rb') as f:
            files = {'file': (zip_filename, f, 'application/x-zip-compressed')}
            response = requests.post(endpoint_url, files=files, timeout=10)
            
        if response.status_code == 200:
            data = response.json()
            pred = data.get("prediction", "UNKNOWN")
            conf = data.get("confidence", 0.0)
            latest_prediction["text"] = f"MATCH: {pred} (Conf: {conf:.2f})"
            latest_prediction["timestamp"] = time.time()
            print(f"[NETWORK SUCCESS] {latest_prediction['text']}")
            
    except requests.exceptions.RequestException as e:
        print(f"[NETWORK FAILED] Error: {e}")

    try:
        if os.path.exists(zip_filename): os.remove(zip_filename)
        if os.path.exists(video_path): os.remove(video_path)
        if os.path.exists(folder_path): shutil.rmtree(folder_path)
    except Exception as e:
        print(f"[CLEANUP ERROR] {e}")

# ==========================================
# 3. THE THREAD MANAGER (Unchanged)
# ==========================================
def thread_pipeline_manager(video_filename, folder_name, skip_rate, api_url):
    process_video_to_silhouettes_custom(
        video_path=video_filename, output_folder=folder_name,
        img_size=(64, 64), inference_size=720,
        confidence_threshold=0.7, extract_every_n_frame=skip_rate
    )
    send_to_endpoint_and_cleanup(folder_name, video_filename, endpoint_url=api_url)

# ==========================================
# 4. THE MAIN CAMERA ENGINE
# ==========================================
def auto_capture_always_on_threaded_720p():
    global latest_prediction, camera_active
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n[SYSTEM] Benchmarking camera hardware... Please wait 2 seconds.")
    start_time = time.time()
    for _ in range(20): cap.read()
    true_fps = 20 / (time.time() - start_time)
    
    frames_to_capture = int(true_fps * 5.0)    
    dynamic_skip = max(1, round(true_fps / 10.0))
    
    zone_w = int(width * 0.60); zone_h = int(height * 1.0)
    zx1 = int((width - zone_w) / 2)
    zy1 = 0; zx2 = zx1 + zone_w; zy2 = height
    
    print("\n[SYSTEM] Loading fast Watcher model...")
    watcher_model = YOLO("yolov8n.pt")
    
    state = "WAITING"
    frames_captured = 0
    subject_counter = 1
    cooldown_frames = 0
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None; filename = ""

    print("\n[READY] Camera Live. Walk into frame...")

    # MODIFIED: The loop now checks if `camera_active` is True!
    while cap.isOpened() and camera_active:
        ret, frame = cap.read()
        if not ret: break

        display_frame = frame.copy()

        if time.time() - latest_prediction["timestamp"] < 10.0:
            cv2.rectangle(display_frame, (30, 30), (700, 110), (0, 0, 0), -1)
            cv2.putText(display_frame, latest_prediction["text"], (50, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        if state == "WAITING":
            if cooldown_frames > 0:
                cooldown_frames -= 1
                cv2.putText(display_frame, "PROCESSING PREVIOUS...", (zx1 + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                results = watcher_model.predict(frame, classes=[0], verbose=False)
                person_in_zone = False

                for r in results:
                    for box in r.boxes:
                        px1, py1, px2, py2 = map(int, box.xyxy[0])
                        if not (px2 < zx1 or px1 > zx2 or py2 < zy1 or py1 > zy2):
                            person_in_zone = True
                            break

                cv2.rectangle(display_frame, (zx1, zy1), (zx2, zy2), (255, 0, 0), 2)
                cv2.putText(display_frame, "WAITING FOR SUBJECT...", (zx1 + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                if person_in_zone:
                    state = "CAPTURING"
                    frames_captured = 0
                    filename = f"live_subject_{subject_counter}.avi"
                    out = cv2.VideoWriter(filename, fourcc, true_fps, (width, height))

        elif state == "CAPTURING":
            out.write(frame)
            frames_captured += 1
            
            cv2.rectangle(display_frame, (zx1, zy1), (zx2, zy2), (0, 0, 255), 4)
            cv2.putText(display_frame, f"CAPTURING: {frames_captured}/{frames_to_capture}", (zx1 + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            if frames_captured >= frames_to_capture:
                out.release()
                folder_name = f"processed_subject_{subject_counter}"
                my_api_url = "http://140.245.221.62:8000/predict" 
                
                worker = threading.Thread(target=thread_pipeline_manager, args=(filename, folder_name, dynamic_skip, my_api_url))
                worker.start()
                
                subject_counter += 1
                state = "WAITING"
                cooldown_frames = int(true_fps) 

        cv2.imshow("Always-On Biometric Scanner", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            camera_active = False # Safe shutdown if 'q' is pressed

    cap.release()
    if out is not None: out.release()
    cv2.destroyAllWindows()
    print("[SYSTEM] Camera offline. Waiting for next start command...")

# ==========================================
# 5. THE MASTER EXECUTION THREAD
# ==========================================
if __name__ == "__main__":
    # 1. Start the Flask "Ear" on a background thread
    server_thread = threading.Thread(target=run_flask_server, daemon=True)
    server_thread.start()
    
    print("=====================================================")
    print("[BRIDGE ACTIVE] Listening for commands on port 5001")
    print("=====================================================")

    # 2. The Main Thread Loop
    # It just sits completely silently until the web app says "/start_camera"
    try:
        while True:
            if camera_active:
                # When the switch is flipped, boot the heavy AI camera loop!
                # When the web app says stop, this function gracefully finishes and returns here.
                auto_capture_always_on_threaded_720p()
            else:
                time.sleep(1) # Sleep to save CPU power while waiting
                
    except KeyboardInterrupt:
        print("\n[SYSTEM] Shutting down bridge completely.")