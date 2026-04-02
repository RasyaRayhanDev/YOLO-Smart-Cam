from ultralytics import YOLO
import cv2
from time import perf_counter
from collections import defaultdict

class PersonTracker:
    def __init__(self, model_path="yolo11n.pt"):
        self.model = YOLO(model_path)
        self.tracker_time = {}
        self.tracker_first_seen = {}
        
    def format_duration(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"
    
    def process_frame(self, frame):
        results = self.model.track(frame, persist=True, classes=[0], verbose=False)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            current_time = perf_counter()
            active_ids = set()
            
            for box, track_id in zip(boxes, track_ids):
                active_ids.add(track_id)
                
                if track_id not in self.tracker_time:
                    self.tracker_time[track_id] = current_time
                    self.tracker_first_seen[track_id] = current_time
                
                self.tracker_time[track_id] = current_time
                
                duration = current_time - self.tracker_first_seen[track_id]
                time_str = self.format_duration(duration)
                
                x1, y1, x2, y2 = map(int, box)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                label = f"ID: {track_id} | {time_str}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            inactive_ids = set(self.tracker_time.keys()) - active_ids
            for inactive_id in inactive_ids:
                if current_time - self.tracker_time[inactive_id] > 5:
                    del self.tracker_time[inactive_id]
                    del self.tracker_first_seen[inactive_id]
        
        return frame
    
    def run(self, source=0):
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Error: Tidak bisa membuka video source: {source}")
            return
        
        print("Tekan 'q' untuk keluar")
        print("Tekan 's' untuk screenshot")
        
        while cap.isOpened():
            success, frame = cap.read()
            
            if not success:
                print("Video selesai atau error membaca frame")
                break
            
            processed_frame = self.process_frame(frame)
            
            info_text = f"Total Orang Terdeteksi: {len(self.tracker_time)}"
            cv2.putText(processed_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Person Tracker - Time Monitoring", processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"screenshot_{int(perf_counter())}.jpg"
                cv2.imwrite(filename, processed_frame)
                print(f"Screenshot disimpan: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        self.print_summary()
    
    def print_summary(self):
        print("\n=== RINGKASAN TRACKING ===")
        if self.tracker_first_seen:
            for track_id in sorted(self.tracker_first_seen.keys()):
                duration = self.tracker_time[track_id] - self.tracker_first_seen[track_id]
                print(f"ID {track_id}: {self.format_duration(duration)}")
        else:
            print("Tidak ada orang yang terdeteksi")

if __name__ == "__main__":
    tracker = PersonTracker()
    
    print("=== Person Tracker dengan Time Monitoring ===")
    print("Pilih sumber video:")
    print("1. Webcam")
    print("2. File video")
    
    choice = input("Pilihan (1/2): ").strip()
    
    if choice == "1":
        tracker.run(source=0)
    elif choice == "2":
        video_path = input("Masukkan path video: ").strip()
        tracker.run(source=video_path)
    else:
        print("Pilihan tidak valid, menggunakan webcam...")
        tracker.run(source=0)
