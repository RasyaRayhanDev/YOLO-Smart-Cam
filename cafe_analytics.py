import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from deepface import DeepFace
import os
from datetime import datetime, timedelta
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle

class CafeAnalytics:
    def __init__(self, data_file="cafe_data.json"):
        self.data_file = data_file
        self.load_data()
    
    def load_data(self):
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = {
                "daily_visitors": {},
                "visitor_durations": []
            }
    
    def save_data(self):
        with open(self.data_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def add_visitor(self, duration_seconds):
        today = datetime.now().strftime("%Y-%m-%d")
        
        if today not in self.data["daily_visitors"]:
            self.data["daily_visitors"][today] = 0
        
        self.data["daily_visitors"][today] += 1
        
        self.data["visitor_durations"].append({
            "date": today,
            "duration_minutes": duration_seconds / 60,
            "person_id": None
        })
        
        self.save_data()
    
    def update_active_visitor(self, person_id, duration_seconds):
        today = datetime.now().strftime("%Y-%m-%d")
        
        for visitor in self.data["visitor_durations"]:
            if visitor.get("person_id") == person_id and visitor["date"] == today:
                visitor["duration_minutes"] = duration_seconds / 60
                self.save_data()
                return
        
        if today not in self.data["daily_visitors"]:
            self.data["daily_visitors"][today] = 0
        
        self.data["daily_visitors"][today] += 1
        
        self.data["visitor_durations"].append({
            "date": today,
            "duration_minutes": duration_seconds / 60,
            "person_id": person_id
        })
        
        self.save_data()
    
    def get_average_visitors_per_day(self, days=7):
        if not self.data["daily_visitors"]:
            return 0
        
        recent_dates = sorted(self.data["daily_visitors"].keys())[-days:]
        total = sum(self.data["daily_visitors"][date] for date in recent_dates)
        return total / len(recent_dates) if recent_dates else 0
    
    def get_average_duration(self, days=7):
        if not self.data["visitor_durations"]:
            return 0
        
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        recent_durations = [
            v["duration_minutes"] for v in self.data["visitor_durations"]
            if v["date"] >= cutoff_date
        ]
        
        return sum(recent_durations) / len(recent_durations) if recent_durations else 0
    
    def get_daily_chart_data(self, days=7):
        if not self.data["daily_visitors"]:
            return pd.DataFrame()
        
        dates = sorted(self.data["daily_visitors"].keys())[-days:]
        visitors = [self.data["daily_visitors"][date] for date in dates]
        
        return pd.DataFrame({
            "Tanggal": dates,
            "Pengunjung": visitors
        })
    
    def get_duration_distribution(self):
        if not self.data["visitor_durations"]:
            return pd.DataFrame()
        
        durations = [v["duration_minutes"] for v in self.data["visitor_durations"]]
        return pd.DataFrame({"Durasi (menit)": durations})

class CafeTrackerStreamlit:
    def __init__(self, model_path="yolo26n.pt", pose_model_path="yolo26n-pose.pt", similarity_threshold=0.65):
        self.model = YOLO(model_path)
        self.pose_model = YOLO(pose_model_path)
        self.similarity_threshold = similarity_threshold
        
        self.person_database = {}
        self.tracker_time = {}
        self.tracker_first_seen = {}
        self.person_counter = 0
        self.completed_visitors = []
        self.person_colors = {}
        self.person_activities = {}
        self.last_json_update = 0
        
        self.temp_face_dir = "temp_faces"
        os.makedirs(self.temp_face_dir, exist_ok=True)
        
        self.analytics = CafeAnalytics()
        self.embeddings_file = "person_embeddings.pkl"
        
        self.load_embeddings()
    
    def load_embeddings(self):
        if os.path.exists(self.embeddings_file):
            try:
                with open(self.embeddings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.person_database = data.get('person_database', {})
                    self.person_counter = data.get('person_counter', 0)
                    self.person_colors = data.get('person_colors', {})
                print(f"✅ Loaded {len(self.person_database)} persons from database")
            except:
                print("⚠️ Could not load embeddings, starting fresh")
    
    def save_embeddings(self):
        try:
            data = {
                'person_database': self.person_database,
                'person_counter': self.person_counter,
                'person_colors': self.person_colors
            }
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"💾 Saved {len(self.person_database)} persons to database")
        except Exception as e:
            print(f"⚠️ Could not save embeddings: {e}")
    
    def extract_color_histogram(self, frame, box):
        try:
            x1, y1, x2, y2 = map(int, box)
            h, w = frame.shape[:0]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                return None
            
            hist = cv2.calcHist([person_crop], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            return hist
        except:
            return None
    
    def find_matching_by_color(self, color_hist):
        if color_hist is None:
            return None
        
        best_match_id = None
        best_similarity = 0
        
        for person_id, hist in self.person_colors.items():
            similarity = cv2.compareHist(color_hist, hist, cv2.HISTCMP_CORREL)
            
            if similarity > best_similarity and similarity > 0.75:
                best_similarity = similarity
                best_match_id = person_id
        
        return best_match_id
    
    def extract_face_embedding(self, frame, box):
        try:
            x1, y1, x2, y2 = map(int, box)
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            box_width = x2 - x1
            box_height = y2 - y1
            
            if box_width < 60 or box_height < 60:
                return None
            
            person_crop = frame[y1:y2, x1:x2]
            
            if person_crop.size == 0:
                return None
            
            temp_path = os.path.join(self.temp_face_dir, f"temp_face_{datetime.now().timestamp()}.jpg")
            cv2.imwrite(temp_path, person_crop)
            
            try:
                embedding = DeepFace.represent(
                    img_path=temp_path,
                    model_name="Facenet",
                    enforce_detection=False,
                    detector_backend="opencv"
                )
                
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                if embedding and len(embedding) > 0:
                    return np.array(embedding[0]["embedding"])
                return None
            except:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return None
        except Exception as e:
            return None
    
    def find_matching_person(self, embedding):
        if embedding is None:
            return None
        
        best_match_id = None
        best_similarity = 0
        
        for person_id, data in self.person_database.items():
            embeddings_list = data.get("embeddings", [data.get("embedding")])
            
            max_similarity = 0
            for stored_embedding in embeddings_list:
                if stored_embedding is None:
                    continue
                    
                similarity = np.dot(embedding, stored_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(stored_embedding)
                )
                max_similarity = max(max_similarity, similarity)
            
            adaptive_threshold = self.similarity_threshold
            if len(embeddings_list) >= 2:
                adaptive_threshold -= 0.08
            elif len(embeddings_list) >= 4:
                adaptive_threshold -= 0.12
            
            if max_similarity > best_similarity and max_similarity > adaptive_threshold:
                best_similarity = max_similarity
                best_match_id = person_id
        
        if best_match_id:
            print(f"🎯 Match found: similarity={best_similarity:.3f}, threshold={adaptive_threshold:.3f}")
        
        return best_match_id
    
    def classify_activity(self, keypoints):
        if keypoints is None or len(keypoints) == 0:
            return "Unknown"
        
        try:
            kpts = keypoints[0]
            
            nose = kpts[0]
            left_shoulder = kpts[5]
            right_shoulder = kpts[6]
            left_hip = kpts[11]
            right_hip = kpts[12]
            left_knee = kpts[13]
            right_knee = kpts[14]
            left_ankle = kpts[15]
            right_ankle = kpts[16]
            
            if nose[2] < 0.3 or left_hip[2] < 0.3 or right_hip[2] < 0.3:
                return "Unknown"
            
            shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
            hip_y = (left_hip[1] + right_hip[1]) / 2
            knee_y = (left_knee[1] + right_knee[1]) / 2 if left_knee[2] > 0.3 and right_knee[2] > 0.3 else hip_y
            ankle_y = (left_ankle[1] + right_ankle[1]) / 2 if left_ankle[2] > 0.3 and right_ankle[2] > 0.3 else knee_y
            
            torso_length = hip_y - shoulder_y
            if torso_length <= 0:
                return "Unknown"
            
            leg_bend_ratio = (ankle_y - hip_y) / torso_length
            
            if leg_bend_ratio < 0.5:
                return "Sitting"
            elif leg_bend_ratio < 1.2:
                return "Standing"
            else:
                return "Walking"
                
        except Exception as e:
            return "Unknown"
    
    def format_duration(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"
    
    def process_frame(self, frame):
        results = self.model.track(frame, persist=True, classes=[0], verbose=False)
        pose_results = self.pose_model(frame, verbose=False)
        current_time = datetime.now().timestamp()
        
        active_persons = set()
        
        pose_data = {}
        if pose_results and len(pose_results) > 0 and pose_results[0].keypoints is not None:
            for idx, kpts in enumerate(pose_results[0].keypoints.data.cpu().numpy()):
                pose_data[idx] = kpts
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            for idx, (box, track_id) in enumerate(zip(boxes, track_ids)):
                activity = "Unknown"
                if idx in pose_data:
                    activity = self.classify_activity([pose_data[idx]])
                
                if track_id not in self.tracker_time:
                    embedding = self.extract_face_embedding(frame, box)
                    color_hist = self.extract_color_histogram(frame, box)
                    matched_person_id = None
                    
                    if embedding is not None:
                        matched_person_id = self.find_matching_person(embedding)
                        if matched_person_id:
                            print(f"✅ ReID by Face: Track {track_id} → Person {matched_person_id}")
                    
                    if matched_person_id is None and color_hist is not None:
                        matched_person_id = self.find_matching_by_color(color_hist)
                        if matched_person_id:
                            print(f"✅ ReID by Color: Track {track_id} → Person {matched_person_id}")
                    
                    if matched_person_id is not None:
                        person_id = matched_person_id
                        
                        if embedding is not None:
                            if "embeddings" not in self.person_database[person_id]:
                                self.person_database[person_id]["embeddings"] = [
                                    self.person_database[person_id].get("embedding")
                                ]
                            
                            if len(self.person_database[person_id]["embeddings"]) < 5:
                                self.person_database[person_id]["embeddings"].append(embedding)
                                self.save_embeddings()
                                print(f"📸 Added embedding #{len(self.person_database[person_id]['embeddings'])} for Person {person_id}")
                        
                        if color_hist is not None:
                            self.person_colors[person_id] = color_hist
                    else:
                        self.person_counter += 1
                        person_id = self.person_counter
                        
                        if embedding is not None:
                            self.person_database[person_id] = {
                                "embeddings": [embedding],
                                "first_seen": current_time
                            }
                        else:
                            self.person_database[person_id] = {
                                "embeddings": [],
                                "first_seen": current_time
                            }
                        
                        if color_hist is not None:
                            self.person_colors[person_id] = color_hist
                        
                        self.tracker_first_seen[person_id] = current_time
                        print(f"🆕 New person: Person {person_id}")
                        
                        self.save_embeddings()
                    
                    self.tracker_time[track_id] = {
                        "person_id": person_id,
                        "last_seen": current_time
                    }
                
                person_id = self.tracker_time[track_id]["person_id"]
                self.tracker_time[track_id]["last_seen"] = current_time
                self.person_activities[person_id] = activity
                active_persons.add(person_id)
                
                if person_id in self.tracker_first_seen:
                    duration = current_time - self.tracker_first_seen[person_id]
                else:
                    duration = current_time - self.person_database[person_id]["first_seen"]
                
                time_str = self.format_duration(duration)
                x1, y1, x2, y2 = map(int, box)
                
                activity_color = {
                    "Sitting": (255, 165, 0),
                    "Standing": (0, 255, 0),
                    "Walking": (0, 191, 255),
                    "Unknown": (128, 128, 128)
                }.get(activity, (128, 128, 128))
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), activity_color, 2)
                label = f"Person {person_id} | {activity} | {time_str}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), activity_color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        inactive_tracks = []
        for track_id, data in self.tracker_time.items():
            if current_time - data["last_seen"] > 10:
                person_id = data["person_id"]
                if person_id not in active_persons and person_id not in self.completed_visitors:
                    if person_id in self.tracker_first_seen:
                        total_duration = current_time - self.tracker_first_seen[person_id]
                    else:
                        total_duration = current_time - self.person_database[person_id]["first_seen"]
                    
                    self.analytics.add_visitor(total_duration)
                    self.completed_visitors.append(person_id)
                    self.save_embeddings()
                
                inactive_tracks.append(track_id)
        
        for track_id in inactive_tracks:
            del self.tracker_time[track_id]
        
        if current_time - self.last_json_update > 5:
            for person_id in active_persons:
                if person_id in self.tracker_first_seen:
                    duration = current_time - self.tracker_first_seen[person_id]
                else:
                    duration = current_time - self.person_database[person_id]["first_seen"]
                
                self.analytics.update_active_visitor(person_id, duration)
            
            self.last_json_update = current_time
        
        return frame

def main():
    st.set_page_config(page_title="Cafe Analytics Dashboard", layout="wide")
    
    st.title("☕ Cafe Analytics Dashboard")
    
    analytics = CafeAnalytics()
    is_tracking = "tracker" in st.session_state and st.session_state.get("tracking_active", False)
    
    tab1, tab2 = st.tabs(["📊 Analytics", "📹 Live Tracking"])
    
    with tab1:
        if is_tracking:
            st.success("🔴 LIVE - Data di-update otomatis setiap 2 detik")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_visitors_7d = analytics.get_average_visitors_per_day(7)
            st.metric("Rata-rata Pengunjung/Hari (7 hari)", f"{avg_visitors_7d:.1f}")
        
        with col2:
            all_durations_seconds = [v["duration_minutes"] * 60 for v in analytics.data["visitor_durations"]]
            
            if is_tracking and "tracker" in st.session_state:
                tracker = st.session_state.tracker
                current_time = datetime.now().timestamp()
                
                for person_id in set(data["person_id"] for data in tracker.tracker_time.values()):
                    if person_id in tracker.tracker_first_seen:
                        duration_seconds = current_time - tracker.tracker_first_seen[person_id]
                        all_durations_seconds.append(duration_seconds)
                    elif person_id in tracker.person_database:
                        duration_seconds = current_time - tracker.person_database[person_id]["first_seen"]
                        all_durations_seconds.append(duration_seconds)
            
            avg_duration_seconds = 0.0
            if all_durations_seconds:
                avg_duration_seconds = sum(all_durations_seconds) / len(all_durations_seconds)
            
            avg_hours = int(avg_duration_seconds // 3600)
            avg_mins = int((avg_duration_seconds % 3600) // 60)
            avg_secs = int(avg_duration_seconds % 60)
            
            if avg_hours > 0:
                duration_str = f"{avg_hours}h {avg_mins}m {avg_secs}s"
            elif avg_mins > 0:
                duration_str = f"{avg_mins}m {avg_secs}s"
            else:
                duration_str = f"{avg_secs}s"
            
            st.metric("Rata-rata Waktu di Cafe (7 hari)", duration_str)
        
        with col3:
            total_visitors = sum(analytics.data["daily_visitors"].values())
            if is_tracking and "tracker" in st.session_state:
                total_visitors = max(total_visitors, st.session_state.tracker.person_counter)
            st.metric("Total Pengunjung", total_visitors)
        
        st.markdown("---")
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.subheader("📈 Pengunjung Harian (7 Hari Terakhir)")
            daily_data = analytics.get_daily_chart_data(7)
            
            if is_tracking and "tracker" in st.session_state:
                today = datetime.now().strftime("%Y-%m-%d")
                current_count = st.session_state.tracker.person_counter
                
                if not daily_data.empty and daily_data.iloc[-1]["Tanggal"] == today:
                    daily_data.iloc[-1, daily_data.columns.get_loc("Pengunjung")] = current_count
                elif current_count > 0:
                    new_row = pd.DataFrame({"Tanggal": [today], "Pengunjung": [current_count]})
                    daily_data = pd.concat([daily_data, new_row], ignore_index=True)
            
            if not daily_data.empty:
                fig = px.bar(daily_data, x="Tanggal", y="Pengunjung", 
                            color="Pengunjung", color_continuous_scale="Greens")
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            else:
                st.info("Belum ada data pengunjung")
        
        with col_chart2:
            st.subheader("⏱️ Distribusi Durasi Kunjungan")
            duration_data = analytics.get_duration_distribution()
            
            if is_tracking and "tracker" in st.session_state:
                tracker = st.session_state.tracker
                current_time = datetime.now().timestamp()
                live_durations = []
                
                for person_id in set(data["person_id"] for data in tracker.tracker_time.values()):
                    if person_id in tracker.tracker_first_seen:
                        duration_minutes = (current_time - tracker.tracker_first_seen[person_id]) / 60
                        live_durations.append(duration_minutes)
                    elif person_id in tracker.person_database:
                        duration_minutes = (current_time - tracker.person_database[person_id]["first_seen"]) / 60
                        live_durations.append(duration_minutes)
                
                if live_durations:
                    live_df = pd.DataFrame({"Durasi (menit)": live_durations})
                    if not duration_data.empty:
                        duration_data = pd.concat([duration_data, live_df], ignore_index=True)
                    else:
                        duration_data = live_df
            
            if not duration_data.empty:
                fig = px.histogram(duration_data, x="Durasi (menit)", 
                                 nbins=20, color_discrete_sequence=["#2ecc71"])
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            else:
                st.info("Belum ada data durasi")
        
        st.markdown("---")
        st.subheader("📋 Data Detail")
        
        if analytics.data["daily_visitors"] or is_tracking:
            daily_dict = analytics.data["daily_visitors"].copy()
            
            if is_tracking and "tracker" in st.session_state:
                today = datetime.now().strftime("%Y-%m-%d")
                daily_dict[today] = st.session_state.tracker.person_counter
            
            if daily_dict:
                df_daily = pd.DataFrame([
                    {"Tanggal": date, "Jumlah Pengunjung": count}
                    for date, count in sorted(daily_dict.items(), reverse=True)
                ])
                st.dataframe(df_daily, use_container_width=True)
    
    with tab2:
        st.subheader("🎥 Live Tracking dari Webcam")
        
        source = 0
        
        start_tracking = st.button("🚀 Mulai Tracking")
        stop_tracking = st.button("⏹️ Stop Tracking")
        
        if "tracking_active" not in st.session_state:
            st.session_state.tracking_active = False
        
        if start_tracking:
            st.session_state.tracking_active = True
            if "tracker" not in st.session_state:
                st.session_state.tracker = CafeTrackerStreamlit(
                    model_path="yolo26n.pt",
                    pose_model_path="yolo26n-pose.pt",
                    similarity_threshold=0.65
                )
                st.session_state.cap = cv2.VideoCapture(source)
        
        if stop_tracking:
            st.session_state.tracking_active = False
            if "cap" in st.session_state:
                st.session_state.cap.release()
                if "tracker" in st.session_state:
                    st.session_state.tracker.save_embeddings()
                del st.session_state.cap
                del st.session_state.tracker
        
        if st.session_state.tracking_active:
            tracker = st.session_state.tracker
            cap = st.session_state.cap
            
            status_placeholder = st.empty()
            status_placeholder.info("🔴 Tracking aktif... Tekan 'Stop Tracking' untuk berhenti")
            
            stframe = st.empty()
            metrics_placeholder = st.empty()
            
            frame_count = 0
            max_frames = 60
            
            while st.session_state.tracking_active and frame_count < max_frames:
                success, frame = cap.read()
                if not success:
                    st.error("Tidak bisa membaca frame dari webcam")
                    st.session_state.tracking_active = False
                    break
                
                processed_frame = tracker.process_frame(frame)
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                stframe.image(processed_frame, channels="RGB", use_container_width=True)
                
                with metrics_placeholder.container():
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Pengunjung Saat Ini", len(set(data["person_id"] for data in tracker.tracker_time.values())))
                    col2.metric("Total Pengunjung Hari Ini", tracker.person_counter)
                    
                    activity_counts = {}
                    for activity in tracker.person_activities.values():
                        activity_counts[activity] = activity_counts.get(activity, 0) + 1
                    activity_str = " | ".join([f"{act}: {count}" for act, count in activity_counts.items()])
                    col3.metric("Aktivitas", activity_str if activity_str else "N/A")
                
                frame_count += 1
                import time
                time.sleep(0.033)
            
            if frame_count >= max_frames and st.session_state.tracking_active:
                st.rerun()
        else:
            st.info("Klik 'Mulai Tracking' untuk memulai")

if __name__ == "__main__":
    main()
