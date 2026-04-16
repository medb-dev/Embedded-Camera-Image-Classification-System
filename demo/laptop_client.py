import cv2
import requests
import threading
import time
from queue import Queue

# ================= CONFIGURATION =================
API_URL = "http://127.0.0.1:8000/api/v1/predict" 
CAMERA_INDEX = 0  # 0 is usually the integrated laptop camera
SEND_INTERVAL = 0.5  # Send a frame every 0.5 seconds (Adjust this for speed)
# =================================================

class LaptopCameraClient:
    def __init__(self):
        # Initialize camera
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        
        # Check if camera opened successfully
        if not self.cap.isOpened():
            print("❌ Error: Could not open laptop camera. Is it being used by another app?")
            exit()

        # Use a Queue to pass frames between the camera thread and the API thread
        self.frame_queue = Queue(maxsize=1)
        self.prediction_text = "Initializing..."
        self.confidence_text = ""
        self.running = True

    def api_worker(self):
        """ 
        This function runs in a separate thread. 
        It waits for a frame to appear in the queue, sends it to the API, 
        and updates the global prediction text.
        """
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                
                try:
                    # Encode the frame as a JPEG image
                    # We compress it to 70% quality to make the request faster
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
                    success, buffer = cv2.imencode('.jpg', frame, encode_param)
                    
                    if success:
                        # The API expects a 'file' field (multipart/form-data)
                        files = {'file': ('image.jpg', buffer.tobytes(), 'image/jpeg')}
                        
                        response = requests.post(API_URL, files=files, timeout=1.5)
                        
                        if response.status_code == 200:
                            data = response.json()
                            self.prediction_text = data['class']
                            self.confidence_text = f"{data['confidence']*100:.1f}%"
                        else:
                            self.prediction_text = "API Error"
                except Exception as e:
                    self.prediction_text = "Connection Error"
                
                # Clear the queue so we don't process old frames
                while not self.frame_queue.empty():
                    self.frame_queue.get()

    def start(self):
        # Start the background API thread
        worker = threading.Thread(target=self.api_worker, daemon=True)
        worker.start()

        last_send_time = 0
        print("📸 Laptop Camera Active. Press 'q' to exit.")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break

            # --- Optimization ---
            # Create a smaller version of the frame to send to API (saves bandwidth)
            # Your API resizes it to 32x32 anyway, so sending a huge 1080p image is wasteful.
            small_frame = cv2.resize(frame, (320, 240))

            # Send frame to API thread based on the SEND_INTERVAL
            current_time = time.time()
            if current_time - last_send_time > SEND_INTERVAL:
                if self.frame_queue.empty():
                    self.frame_queue.put(small_frame)
                    last_send_time = current_time

            # --- UI Overlay ---
            # Draw a rectangle for the result box
            cv2.rectangle(frame, (10, 10), (350, 70), (0, 0, 0), -1)
            
            # Display Prediction and Confidence
            full_text = f"Prediction: {self.prediction_text} | Conf: {self.confidence_text}"
            cv2.putText(frame, full_text, (20, 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Embedded Vision Client", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    client = LaptopCameraClient()
    client.start()
