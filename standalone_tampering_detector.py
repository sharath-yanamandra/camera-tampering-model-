
import cv2
import numpy as np
import time
import threading
import queue
from collections import deque
from datetime import datetime
import os
import json
import sys

# Try to import PIL for Windows display
try:
    from PIL import Image, ImageTk
    import tkinter as tk
    from tkinter import ttk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# RTSP Configuration - Edit these as needed
RTSP_URL = "rtsp://admin:administrator@192.168.29.74:554/ch0_0.264"
RTSP_BACKUP_URL = None  # Optional backup camera
RTSP_TIMEOUT = 10  # seconds

class HistogramQueue:
    """Circular buffer for storing histogram data"""
    
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.queue = deque(maxlen=maxsize)
        
    def enqueue(self, item):
        self.queue.append(item)
        
    def dequeue(self):
        if self.queue:
            return self.queue.popleft()
        return None
        
    def is_full(self):
        return len(self.queue) == self.maxsize
        
    def is_empty(self):
        return len(self.queue) == 0
        
    def front(self):
        if self.queue:
            return self.queue[0]
        return None
        
    def __len__(self):
        return len(self.queue)
        
    def __iter__(self):
        return iter(self.queue)

class SimpleDisplayManager:
    """Simple display manager using OpenCV - more reliable"""
    
    def __init__(self, window_name="Camera Tampering Detection"):
        self.window_name = window_name
        self.running = False
        
    def start_display(self, width=640, height=480):
        """Start the display window"""
        self.running = True
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        print(f"🖼️  Display window opened: {self.window_name}")
        
    def update_frame(self, frame, status_text=""):
        """Update the frame to display"""
        if self.running:
            # Add status text to frame
            if status_text:
                cv2.putText(frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow(self.window_name, frame)
            
            # Handle window events
            key = cv2.waitKey(1) & 0xFF
            return key
        return -1
    
    def is_running(self):
        """Check if display is still running"""
        return self.running
    
    def close(self):
        """Close the display"""
        self.running = False
        cv2.destroyAllWindows()
        print("🪟 Display window closed")

class TkinterDisplayManager:
    """Advanced tkinter display manager"""
    
    def __init__(self):
        self.root = None
        self.canvas = None
        self.photo = None
        self.running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.frame_queue = queue.Queue(maxsize=2)
        
    def start_display(self, width=640, height=480):
        """Start the display window in main thread"""
        try:
            self.root = tk.Tk()
            self.root.title("Camera Tampering Detection - RTSP Monitor")
            self.root.geometry(f"{width}x{height+120}")
            self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
            
            # Create canvas for image
            self.canvas = tk.Canvas(self.root, width=width, height=height, bg='black')
            self.canvas.pack(pady=10)
            
            # Status text
            self.status_var = tk.StringVar()
            self.status_var.set("Connecting to RTSP stream...")
            status_label = tk.Label(self.root, textvariable=self.status_var, 
                                  font=('Arial', 12), fg='blue')
            status_label.pack()
            
            # RTSP info
            rtsp_display = RTSP_URL.split('@')[1] if '@' in RTSP_URL else RTSP_URL
            rtsp_label = tk.Label(self.root, text=f"Stream: {rtsp_display}", 
                                font=('Arial', 8), fg='gray')
            rtsp_label.pack()
            
            # Control buttons
            button_frame = tk.Frame(self.root)
            button_frame.pack(pady=5)
            
            self.quit_btn = tk.Button(button_frame, text="Quit", 
                                    command=self._quit_callback, 
                                    bg='red', fg='white', width=10)
            self.quit_btn.pack(side=tk.LEFT, padx=5)
            
            self.reset_btn = tk.Button(button_frame, text="Reset", 
                                     command=self._reset_callback, 
                                     bg='orange', fg='white', width=10)
            self.reset_btn.pack(side=tk.LEFT, padx=5)
            
            self.stats_btn = tk.Button(button_frame, text="Save Stats", 
                                     command=self._stats_callback, 
                                     bg='green', fg='white', width=10)
            self.stats_btn.pack(side=tk.LEFT, padx=5)
            
            self.running = True
            
            # Start update loop
            self._update_display()
            
            print("✅ Tkinter GUI initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize tkinter GUI: {e}")
            self.running = False
            raise
    
    def _update_display(self):
        """Update the display with current frame"""
        if self.running and self.root:
            try:
                # Get latest frame from queue
                while not self.frame_queue.empty():
                    try:
                        frame_data = self.frame_queue.get_nowait()
                        self.current_frame = frame_data['frame']
                        if frame_data['status']:
                            self.status_var.set(frame_data['status'])
                    except queue.Empty:
                        break
                
                # Update display
                if self.current_frame is not None:
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                    
                    # Convert to PIL Image
                    pil_image = Image.fromarray(rgb_frame)
                    
                    # Resize to fit canvas
                    canvas_width = self.canvas.winfo_width()
                    canvas_height = self.canvas.winfo_height()
                    if canvas_width > 1 and canvas_height > 1:
                        pil_image = pil_image.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
                    
                    # Convert to PhotoImage
                    self.photo = ImageTk.PhotoImage(pil_image)
                    
                    # Update canvas
                    self.canvas.delete("all")
                    self.canvas.create_image(canvas_width//2, canvas_height//2, image=self.photo)
                
            except Exception as e:
                print(f"Display update error: {e}")
            
            # Schedule next update
            if self.running:
                self.root.after(50, self._update_display)  # 20 FPS
    
    def update_frame(self, frame, status_text=""):
        """Update the frame to display"""
        if self.running:
            try:
                # Add to queue (non-blocking)
                frame_data = {
                    'frame': frame.copy(),
                    'status': status_text
                }
                
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()  # Remove old frame
                    except queue.Empty:
                        pass
                
                self.frame_queue.put_nowait(frame_data)
                
            except queue.Full:
                pass  # Skip frame if queue is full
            except Exception as e:
                print(f"Frame update error: {e}")
    
    def _on_closing(self):
        """Handle window closing"""
        self.running = False
        if self.root:
            self.root.destroy()
    
    def _quit_callback(self):
        """Handle quit button"""
        print("🛑 Quit requested via GUI")
        self.running = False
        global gui_quit_requested
        gui_quit_requested = True
    
    def _reset_callback(self):
        """Handle reset button"""
        print("🔄 Reset requested via GUI")
        global gui_reset_requested
        gui_reset_requested = True
    
    def _stats_callback(self):
        """Handle stats button"""
        print("📊 Stats requested via GUI")
        global gui_stats_requested
        gui_stats_requested = True
    
    def is_running(self):
        """Check if display is still running"""
        return self.running
    
    def close(self):
        """Close the display"""
        self.running = False
        if self.root:
            try:
                self.root.quit()
                self.root.destroy()
            except:
                pass
    
    def process_events(self):
        """Process tkinter events"""
        if self.root and self.running:
            try:
                self.root.update_idletasks()
                self.root.update()
            except tk.TclError:
                self.running = False

class CameraTamperingDetector:
    """Standalone camera tampering detector"""
    
    def __init__(self, 
                 img_width=300, 
                 img_height=300,
                 chroma_threshold=50.0,
                 gradient_threshold=30.0,
                 combined_threshold=40.0):
        
        # Processing parameters
        self.img_width = img_width
        self.img_height = img_height
        self.st_pool_size = 15  # Short-term pool size
        self.lt_pool_size = 30  # Long-term pool size
        self.lt_pool_cnt = 3    # Long-term pool count interval
        self.num_bins = 16
        
        # Thresholds
        self.chroma_threshold = chroma_threshold
        self.gradient_threshold = gradient_threshold
        self.combined_threshold = combined_threshold
        
        # Initialize queues
        self.st_frame_queue = HistogramQueue(self.st_pool_size)
        self.lt_frame_queue = HistogramQueue(self.lt_pool_size)
        self.chroma_hist_queue = HistogramQueue(self.lt_pool_size)
        self.gradient_hist_queue = HistogramQueue(self.lt_pool_size)
        self.cc_queue = HistogramQueue(self.st_pool_size)
        self.gc_queue = HistogramQueue(self.st_pool_size)
        
        # State flags
        self.cc_initialized = False
        self.gc_initialized = False
        self.counter = 0
        
        # Statistics
        self.stats = {
            "frames_processed": 0,
            "tampering_detections": 0,
            "chroma_violations": 0,
            "gradient_violations": 0,
            "combined_violations": 0,
            "average_chroma_diff": 0.0,
            "average_gradient_diff": 0.0,
            "start_time": time.time(),
            "last_detection_time": None
        }
        
        # Recent measurements
        self.recent_chroma_diffs = deque(maxlen=10)
        self.recent_gradient_diffs = deque(maxlen=10)
        
        # Alert state
        self.alert_active = False
        self.alert_start_time = None
        self.alert_duration = 3.0
        self.alert_message = ""
        
        print(f"🎯 Tampering detector initialized:")
        print(f"  📐 Processing size: {img_width}x{img_height}")
        print(f"  🎨 Chroma threshold: {chroma_threshold}")
        print(f"  📊 Gradient threshold: {gradient_threshold}")
        print(f"  🔄 Combined threshold: {combined_threshold}")
    
    def reset_detector(self):
        """Reset the detector state"""
        print("🔄 Resetting detector state...")
        
        # Clear all queues
        self.st_frame_queue = HistogramQueue(self.st_pool_size)
        self.lt_frame_queue = HistogramQueue(self.lt_pool_size)
        self.chroma_hist_queue = HistogramQueue(self.lt_pool_size)
        self.gradient_hist_queue = HistogramQueue(self.lt_pool_size)
        self.cc_queue = HistogramQueue(self.st_pool_size)
        self.gc_queue = HistogramQueue(self.st_pool_size)
        
        # Reset flags
        self.cc_initialized = False
        self.gc_initialized = False
        self.counter = 0
        
        # Reset alert state
        self.alert_active = False
        self.alert_start_time = None
        self.alert_message = ""
        
        # Clear recent measurements
        self.recent_chroma_diffs.clear()
        self.recent_gradient_diffs.clear()
        
        print("✅ Detector reset complete")
    
    def resize_frame(self, frame):
        """Resize frame to standard processing size"""
        return cv2.resize(frame, (self.img_width, self.img_height))
    
    def calc_chroma_histogram(self, frame):
        """Calculate 2D chromaticity histogram"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        return hist
    
    def calc_gradient_histogram(self, frame):
        """Calculate gradient direction histogram"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        normalized = gray.astype(np.float32) / 255.0
        
        grad_x = cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3)
        
        magnitude, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
        hist = cv2.calcHist([angle], [0], None, [self.num_bins], [0, 360])
        
        return hist
    
    def calculate_histogram_diff(self, hist1, hist2):
        """Calculate absolute difference between two histograms"""
        diff = cv2.absdiff(hist1, hist2)
        return np.sum(diff)
    
    def chromaticity_diff(self, current_frame):
        """Calculate chromaticity difference"""
        if not self.cc_initialized:
            print("🎨 Initializing chromaticity comparison...")
            
            for frame in self.lt_frame_queue:
                chroma_hist = self.calc_chroma_histogram(frame)
                self.chroma_hist_queue.enqueue(chroma_hist)
            
            for frame in self.st_frame_queue:
                current_hist = self.calc_chroma_histogram(frame)
                diffs = []
                
                for lt_hist in self.chroma_hist_queue:
                    diff = self.calculate_histogram_diff(current_hist, lt_hist)
                    diffs.append(diff)
                
                avg_diff = np.mean(diffs) if diffs else 0.0
                self.cc_queue.enqueue(avg_diff)
            
            self.cc_initialized = True
            print("✅ Chromaticity initialization complete")
        
        current_hist = self.calc_chroma_histogram(current_frame)
        diffs = []
        
        for lt_hist in self.chroma_hist_queue:
            diff = self.calculate_histogram_diff(current_hist, lt_hist)
            diffs.append(diff)
        
        avg_diff = np.mean(diffs) if diffs else 0.0
        
        if self.cc_queue.is_full():
            self.cc_queue.dequeue()
        self.cc_queue.enqueue(avg_diff)
        
        overall_avg = np.mean(list(self.cc_queue)) if len(self.cc_queue) > 0 else 0.0
        return overall_avg
    
    def gradient_direction_diff(self, current_frame):
        """Calculate gradient direction difference"""
        if not self.gc_initialized:
            print("📐 Initializing gradient direction comparison...")
            
            for frame in self.lt_frame_queue:
                grad_hist = self.calc_gradient_histogram(frame)
                self.gradient_hist_queue.enqueue(grad_hist)
            
            for frame in self.st_frame_queue:
                current_hist = self.calc_gradient_histogram(frame)
                diffs = []
                
                for lt_hist in self.gradient_hist_queue:
                    diff = self.calculate_histogram_diff(current_hist, lt_hist)
                    diffs.append(diff)
                
                avg_diff = np.mean(diffs) if diffs else 0.0
                self.gc_queue.enqueue(avg_diff)
            
            self.gc_initialized = True
            print("✅ Gradient direction initialization complete")
        
        current_hist = self.calc_gradient_histogram(current_frame)
        diffs = []
        
        for lt_hist in self.gradient_hist_queue:
            diff = self.calculate_histogram_diff(current_hist, lt_hist)
            diffs.append(diff)
        
        avg_diff = np.mean(diffs) if diffs else 0.0
        
        if self.gc_queue.is_full():
            self.gc_queue.dequeue()
        self.gc_queue.enqueue(avg_diff)
        
        overall_avg = np.mean(list(self.gc_queue)) if len(self.gc_queue) > 0 else 0.0
        return overall_avg
    
    def detect_tampering(self, frame):
        """Main tampering detection logic"""
        resized_frame = self.resize_frame(frame)
        
        if self.st_frame_queue.is_full():
            self.st_frame_queue.dequeue()
        self.st_frame_queue.enqueue(resized_frame.copy())
        
        tampering_detected = False
        tampering_type = "none"
        chroma_diff = 0.0
        gradient_diff = 0.0
        
        if self.st_frame_queue.is_full() and self.lt_frame_queue.is_full():
            chroma_diff = self.chromaticity_diff(resized_frame)
            gradient_diff = self.gradient_direction_diff(resized_frame)
            
            self.recent_chroma_diffs.append(chroma_diff)
            self.recent_gradient_diffs.append(gradient_diff)
            
            self.stats["average_chroma_diff"] = np.mean(list(self.recent_chroma_diffs))
            self.stats["average_gradient_diff"] = np.mean(list(self.recent_gradient_diffs))
            
            chroma_violation = chroma_diff > self.chroma_threshold
            gradient_violation = gradient_diff > self.gradient_threshold
            combined_violation = (chroma_diff + gradient_diff) / 2 > self.combined_threshold
            
            if chroma_violation or gradient_violation or combined_violation:
                tampering_detected = True
                
                if chroma_violation and gradient_violation:
                    tampering_type = "severe_tampering"
                elif chroma_violation:
                    tampering_type = "color_obstruction"
                elif gradient_violation:
                    tampering_type = "physical_displacement"
                elif combined_violation:
                    tampering_type = "combined_tampering"
                
                self.stats["tampering_detections"] += 1
                self.stats["last_detection_time"] = datetime.now().isoformat()
                if chroma_violation:
                    self.stats["chroma_violations"] += 1
                if gradient_violation:
                    self.stats["gradient_violations"] += 1
                if combined_violation:
                    self.stats["combined_violations"] += 1
                
                if not self.alert_active:
                    self.alert_active = True
                    self.alert_start_time = time.time()
                    self.alert_message = f"🚨 {tampering_type.replace('_', ' ').upper()}"
                    
                    print(f"\n{'='*70}")
                    print(f"🚨 TAMPERING ALERT: {tampering_type.upper()}")
                    print(f"📊 Chroma: {chroma_diff:.2f}/{self.chroma_threshold}")
                    print(f"📊 Gradient: {gradient_diff:.2f}/{self.gradient_threshold}")
                    print(f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}")
                    print(f"{'='*70}\n")
        
        # Manage long-term pool
        if self.st_frame_queue.is_full():
            self.counter += 1
            
            if self.counter % self.lt_pool_cnt == 0:
                self.counter = 0
                
                lt_frame = self.st_frame_queue.front().copy()
                
                if self.lt_frame_queue.is_full():
                    self.lt_frame_queue.dequeue()
                    if self.chroma_hist_queue.is_full():
                        self.chroma_hist_queue.dequeue()
                    if self.gradient_hist_queue.is_full():
                        self.gradient_hist_queue.dequeue()
                
                self.lt_frame_queue.enqueue(lt_frame)
                
                chroma_hist = self.calc_chroma_histogram(lt_frame)
                self.chroma_hist_queue.enqueue(chroma_hist)
                
                grad_hist = self.calc_gradient_histogram(lt_frame)
                self.gradient_hist_queue.enqueue(grad_hist)
        
        self.stats["frames_processed"] += 1
        
        return tampering_detected, tampering_type, chroma_diff, gradient_diff
    
    def draw_status_overlay(self, frame, chroma_diff, gradient_diff, tampering_detected, tampering_type):
        """Draw status overlay on frame"""
        current_time = time.time()
        display_frame = frame.copy()
        
        # Alert overlay
        if self.alert_active:
            alert_age = current_time - self.alert_start_time
            if alert_age <= self.alert_duration:
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], 120), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
                
                cv2.putText(display_frame, self.alert_message, 
                           (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                
                if int(alert_age * 4) % 2 == 0:
                    cv2.putText(display_frame, "CAMERA TAMPERING DETECTED", 
                               (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                self.alert_active = False
        
        # Status panel
        panel_height = 120
        cv2.rectangle(display_frame, (0, display_frame.shape[0] - panel_height), 
                     (display_frame.shape[1], display_frame.shape[0]), (0, 0, 0), -1)
        
        y = display_frame.shape[0] - panel_height + 20
        
        # Status
        pool_status = "🔄 Initializing..."
        color = (0, 255, 255)
        if self.st_frame_queue.is_full() and self.lt_frame_queue.is_full():
            if self.cc_initialized and self.gc_initialized:
                pool_status = "✅ Monitoring Active"
                color = (0, 255, 0)
            else:
                pool_status = "⚙️ Calibrating..."
                color = (0, 165, 255)
        
        cv2.putText(display_frame, pool_status, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Measurements
        y += 25
        chroma_color = (0, 0, 255) if chroma_diff > self.chroma_threshold else (255, 255, 255)
        gradient_color = (0, 0, 255) if gradient_diff > self.gradient_threshold else (255, 255, 255)
        
        cv2.putText(display_frame, f"Chroma: {chroma_diff:.1f}/{self.chroma_threshold:.1f}", 
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, chroma_color, 1)
        cv2.putText(display_frame, f"Gradient: {gradient_diff:.1f}/{self.gradient_threshold:.1f}", 
                   (250, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gradient_color, 1)
        
        # Stats
        y += 20
        uptime = time.time() - self.stats["start_time"]
        fps = self.stats["frames_processed"] / max(1, uptime)
        cv2.putText(display_frame, f"Frames: {self.stats['frames_processed']} | FPS: {fps:.1f} | Detections: {self.stats['tampering_detections']}", 
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Pools
        y += 20
        cv2.putText(display_frame, f"ST: {len(self.st_frame_queue)}/{self.st_pool_size} | LT: {len(self.lt_frame_queue)}/{self.lt_pool_size}", 
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Timestamp
        timestamp = datetime.now().strftime('%H:%M:%S')
        cv2.putText(display_frame, timestamp, (display_frame.shape[1] - 100, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return display_frame
    
    def get_stats(self):
        """Get current statistics"""
        uptime = time.time() - self.stats["start_time"]
        fps = self.stats["frames_processed"] / max(1, uptime)
        
        return {
            **self.stats,
            "uptime_seconds": uptime,
            "fps": fps,
            "pool_status": {
                "st_pool_size": len(self.st_frame_queue),
                "lt_pool_size": len(self.lt_frame_queue),
                "cc_initialized": self.cc_initialized,
                "gc_initialized": self.gc_initialized
            }
        }

def create_rtsp_connection(url, timeout=10):
    """Create RTSP connection with proper configuration"""
    print(f"🔌 Connecting to RTSP: {url}")
    
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Test connection
    start_time = time.time()
    connected = False
    
    while (time.time() - start_time) < timeout:
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                connected = True
                break
        time.sleep(0.1)
    
    if not connected:
        cap.release()
        raise Exception(f"Failed to connect to RTSP stream: {url}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"✅ RTSP connected: {width}x{height} @ {fps} FPS")
    return cap

def save_detection_log(detector, output_dir="tampering_logs"):
    """Save detection statistics to file"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"tampering_log_{timestamp}.json")
    
    stats = detector.get_stats()
    stats["timestamp"] = datetime.now().isoformat()
    stats["rtsp_url"] = RTSP_URL
    
    with open(log_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"📄 Log saved: {log_file}")
    return log_file

# Global GUI communication flags
gui_quit_requested = False
gui_reset_requested = False
gui_stats_requested = False

def main():
    """Main application with improved GUI handling"""
    global gui_quit_requested, gui_reset_requested, gui_stats_requested
    
    print("🎯 Camera Tampering Detection System")
    print("=" * 60)
    print(f"📡 RTSP URL: {RTSP_URL}")
    print("=" * 60)
    
    # Detection parameters
    CHROMA_THRESHOLD = 45.0
    GRADIENT_THRESHOLD = 25.0
    COMBINED_THRESHOLD = 35.0
    
    # Choose display method
    USE_TKINTER = PIL_AVAILABLE
    
    if USE_TKINTER:
        print("🖼️  Using Tkinter GUI")
    else:
        print("🖼️  Using OpenCV display (Tkinter not available)")
    
    # Initialize components
    cap = None
    detector = None
    display_manager = None
    
    try:
        # Connect to RTSP
        print("🔌 Establishing RTSP connection...")
        cap = create_rtsp_connection(RTSP_URL, timeout=RTSP_TIMEOUT)
        
        # Initialize detector
        detector = CameraTamperingDetector(
            img_width=300,
            img_height=300,
            chroma_threshold=CHROMA_THRESHOLD,
            gradient_threshold=GRADIENT_THRESHOLD,
            combined_threshold=COMBINED_THRESHOLD
        )
        
        # Initialize display
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        
        if USE_TKINTER:
            display_manager = TkinterDisplayManager()
            display_manager.start_display(width, height)
        else:
            display_manager = SimpleDisplayManager()
            display_manager.start_display(width, height)
        
        # Main loop variables
        frame_count = 0
        start_time = time.time()
        last_stats_time = start_time
        last_console_update = start_time
        connection_lost_time = None
        
        print("🚀 Starting detection loop...")
        print("Controls:")
        if USE_TKINTER:
            print("  - Use GUI buttons for control")
        else:
            print("  - Press 'q' to quit")
            print("  - Press 's' to save stats")
            print("  - Press 'r' to reset detector")
        print("  - Press Ctrl+C to force quit")
        print("-" * 60)
        
        # Main processing loop
        while True:
            current_time = time.time()
            
            # Handle GUI requests
            if gui_quit_requested:
                print("🛑 Quit requested from GUI")
                break
            
            if gui_reset_requested:
                gui_reset_requested = False
                detector.reset_detector()
            
            if gui_stats_requested:
                gui_stats_requested = False
                try:
                    save_detection_log(detector)
                except Exception as e:
                    print(f"❌ Failed to save stats: {e}")
            
            # Process GUI events (for tkinter)
            if USE_TKINTER and isinstance(display_manager, TkinterDisplayManager):
                if not display_manager.is_running():
                    print("🪟 GUI window closed")
                    break
                try:
                    display_manager.process_events()
                except Exception as e:
                    print(f"GUI error: {e}")
                    break
            
            # Read frame
            ret, frame = cap.read()
            
            if not ret or frame is None:
                if connection_lost_time is None:
                    connection_lost_time = current_time
                    print("⚠️  RTSP connection lost, attempting reconnect...")
                
                # Try reconnect after 2 seconds
                if (current_time - connection_lost_time) > 2.0:
                    try:
                        cap.release()
                        cap = create_rtsp_connection(RTSP_URL, timeout=5)
                        connection_lost_time = None
                        print("✅ RTSP reconnected!")
                        continue
                    except Exception as e:
                        print(f"❌ Reconnect failed: {e}")
                        connection_lost_time = current_time
                
                # Show no signal frame
                if display_manager and display_manager.is_running():
                    no_signal_frame = np.zeros((height, width, 3), dtype=np.uint8)
                    cv2.putText(no_signal_frame, "NO SIGNAL - RECONNECTING...", 
                               (width//2 - 200, height//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    if USE_TKINTER:
                        display_manager.update_frame(no_signal_frame, "Connection Lost - Reconnecting...")
                    else:
                        key = display_manager.update_frame(no_signal_frame, "Connection Lost")
                        if key == ord('q'):
                            break
                
                time.sleep(0.1)
                continue
            
            # Reset connection lost time
            if connection_lost_time is not None:
                connection_lost_time = None
                print("✅ RTSP connection restored")
            
            frame_count += 1
            
            # Detect tampering
            tampering_detected, tampering_type, chroma_diff, gradient_diff = detector.detect_tampering(frame)
            
            # Create display frame
            display_frame = detector.draw_status_overlay(
                frame, chroma_diff, gradient_diff, tampering_detected, tampering_type
            )
            
            # Update display
            if display_manager and display_manager.is_running():
                status_text = f"Frame {frame_count} | "
                if tampering_detected:
                    status_text += f"🚨 {tampering_type.replace('_', ' ').upper()}"
                else:
                    status_text += "✅ Normal"
                
                if USE_TKINTER:
                    display_manager.update_frame(display_frame, status_text)
                else:
                    key = display_manager.update_frame(display_frame, status_text)
                    
                    # Handle OpenCV key events
                    if key == ord('q'):
                        print("👋 Quit requested")
                        break
                    elif key == ord('s'):
                        try:
                            save_detection_log(detector)
                        except Exception as e:
                            print(f"❌ Failed to save stats: {e}")
                    elif key == ord('r'):
                        print("🔄 Reset requested")
                        detector.reset_detector()
            
            # Console updates (every 3 seconds)
            if (current_time - last_console_update) >= 3.0:
                stats = detector.get_stats()
                uptime = current_time - start_time
                fps = frame_count / max(1, uptime)
                
                # Status line
                status_line = f"📊 Frame: {frame_count:6d} | FPS: {fps:5.1f} | "
                status_line += f"Uptime: {uptime:6.1f}s | Detections: {stats['tampering_detections']}"
                print(status_line)
                
                # Detection status
                if tampering_detected:
                    print(f"🚨 ACTIVE ALERT: {tampering_type.replace('_', ' ').upper()}")
                
                # Pool status
                pool_status = "Initializing"
                if detector.st_frame_queue.is_full() and detector.lt_frame_queue.is_full():
                    if detector.cc_initialized and detector.gc_initialized:
                        pool_status = "Active"
                    else:
                        pool_status = "Calibrating"
                
                print(f"📈 Status: {pool_status} | Chroma: {chroma_diff:.1f}/{CHROMA_THRESHOLD} | Gradient: {gradient_diff:.1f}/{GRADIENT_THRESHOLD}")
                
                last_console_update = current_time
            
            # Periodic stats (every 60 seconds)
            if (current_time - last_stats_time) >= 60.0:
                stats = detector.get_stats()
                print(f"\n📈 [{datetime.now().strftime('%H:%M:%S')}] Periodic Summary:")
                print(f"  📺 Frames: {stats['frames_processed']}")
                print(f"  🚨 Detections: {stats['tampering_detections']}")
                print(f"  🎨 Chroma violations: {stats['chroma_violations']}")
                print(f"  📐 Gradient violations: {stats['gradient_violations']}")
                print(f"  ⚡ FPS: {stats['fps']:.2f}")
                
                # Auto-save stats
                try:
                    save_detection_log(detector)
                    print("  💾 Auto-saved stats")
                except Exception as e:
                    print(f"  ❌ Auto-save failed: {e}")
                
                last_stats_time = current_time
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user (Ctrl+C)")
    
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n🧹 Cleanup process...")
        
        # Release video capture
        if cap:
            cap.release()
            print("📹 RTSP connection closed")
        
        # Close display
        if display_manager:
            display_manager.close()
            print("🖼️  Display closed")
        
        # Close OpenCV windows
        cv2.destroyAllWindows()
        
        # Save final stats
        if detector:
            try:
                save_detection_log(detector)
                print("📄 Final stats saved")
            except Exception as e:
                print(f"❌ Failed to save final stats: {e}")
            
            # Print summary
            print("\n" + "="*70)
            print("📊 FINAL SUMMARY")
            print("="*70)
            
            stats = detector.get_stats()
            total_time = time.time() - start_time
            
            print(f"⏱️  Runtime: {total_time:.1f} seconds")
            print(f"📺 Frames: {stats['frames_processed']}")
            print(f"⚡ Avg FPS: {stats['fps']:.2f}")
            print(f"🚨 Total detections: {stats['tampering_detections']}")
            print(f"  🎨 Chroma: {stats['chroma_violations']}")
            print(f"  📐 Gradient: {stats['gradient_violations']}")
            print(f"  🔄 Combined: {stats['combined_violations']}")
            
            if stats['frames_processed'] > 0:
                rate = (stats['tampering_detections'] / stats['frames_processed']) * 100
                print(f"📈 Detection rate: {rate:.2f}%")
            
            print(f"📊 Final measurements:")
            print(f"  🎨 Avg chroma diff: {stats['average_chroma_diff']:.2f}")
            print(f"  📐 Avg gradient diff: {stats['average_gradient_diff']:.2f}")
            
            if stats['last_detection_time']:
                print(f"🕐 Last detection: {stats['last_detection_time']}")
            
            print("="*70)
            print("🎯 Camera Tampering Detection Complete!")

def test_rtsp_connection():
    """Test RTSP connection without full detection"""
    print("🔍 Testing RTSP connection...")
    print(f"📡 URL: {RTSP_URL}")
    
    try:
        cap = create_rtsp_connection(RTSP_URL, timeout=15)
        
        print("📺 Testing frame capture...")
        for i in range(10):
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✅ Frame {i+1}: {frame.shape} - OK")
            else:
                print(f"❌ Frame {i+1}: Failed")
            time.sleep(0.2)
        
        cap.release()
        print("✅ RTSP connection test completed successfully!")
        
    except Exception as e:
        print(f"❌ RTSP connection test failed: {e}")

def show_help():
    """Show help information"""
    print("🎯 Camera Tampering Detection System")
    print("=" * 60)
    print("Usage:")
    print("  python3 tampering_detector.py        - Run full detection")
    print("  python3 tampering_detector.py test   - Test RTSP connection")
    print("  python3 tampering_detector.py help   - Show this help")
    print()
    print("Configuration:")
    print(f"  📡 RTSP URL: {RTSP_URL}")
    print(f"  ⏱️  Timeout: {RTSP_TIMEOUT} seconds")
    print()
    print("Requirements:")
    print("  - OpenCV (cv2)")
    print("  - NumPy")
    print("  - PIL/Pillow (optional, for advanced GUI)")
    print()
    print("Install dependencies:")
    print("  pip install opencv-python numpy pillow")
    print()
    print("Controls (during detection):")
    if PIL_AVAILABLE:
        print("  - GUI buttons for control")
    else:
        print("  - 'q' key to quit")
        print("  - 's' key to save stats")
        print("  - 'r' key to reset detector")
    print("  - Ctrl+C to force quit")
    print("=" * 60)

if __name__ == "__main__":
    # Handle command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test_rtsp_connection()
        elif sys.argv[1] == "help":
            show_help()
        else:
            print("❌ Unknown argument. Use 'test' or 'help'")
            show_help()
    else:
        main()

'''
import cv2
import numpy as np
import time
import argparse
import threading
import queue
from collections import deque
from datetime import datetime
import os
import json
import sys

# Try to import PIL for Windows display
try:
    from PIL import Image, ImageDraw, ImageFont
    import tkinter as tk
    from tkinter import ttk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

class HistogramQueue:
    """Circular buffer for storing histogram data"""
    
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.queue = deque(maxlen=maxsize)
        
    def enqueue(self, item):
        self.queue.append(item)
        
    def dequeue(self):
        if self.queue:
            return self.queue.popleft()
        return None
        
    def is_full(self):
        return len(self.queue) == self.maxsize
        
    def is_empty(self):
        return len(self.queue) == 0
        
    def front(self):
        if self.queue:
            return self.queue[0]
        return None
        
    def __len__(self):
        return len(self.queue)
        
    def __iter__(self):
        return iter(self.queue)

class WindowsDisplayManager:
    """Windows-compatible display manager using tkinter"""
    
    def __init__(self):
        self.root = None
        self.canvas = None
        self.photo = None
        self.running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
    def start_display(self, width=640, height=480):
        """Start the display window in a separate thread"""
        self.display_thread = threading.Thread(target=self._display_worker, args=(width, height))
        self.display_thread.daemon = True
        self.display_thread.start()
        time.sleep(1)  # Give time for window to initialize
        
    def _display_worker(self, width, height):
        """Display worker running in separate thread"""
        try:
            self.root = tk.Tk()
            self.root.title("Camera Tampering Detection")
            self.root.geometry(f"{width}x{height+100}")
            
            # Create canvas for image
            self.canvas = tk.Canvas(self.root, width=width, height=height, bg='black')
            self.canvas.pack(pady=10)
            
            # Status text
            self.status_var = tk.StringVar()
            self.status_var.set("Initializing...")
            status_label = tk.Label(self.root, textvariable=self.status_var, font=('Arial', 12))
            status_label.pack()
            
            # Control buttons
            button_frame = tk.Frame(self.root)
            button_frame.pack(pady=5)
            
            tk.Button(button_frame, text="Quit", command=self._quit_callback).pack(side=tk.LEFT, padx=5)
            tk.Button(button_frame, text="Reset", command=self._reset_callback).pack(side=tk.LEFT, padx=5)
            tk.Button(button_frame, text="Save Stats", command=self._stats_callback).pack(side=tk.LEFT, padx=5)
            
            self.running = True
            
            # Update display periodically
            self.root.after(33, self._update_display)  # ~30 FPS
            self.root.mainloop()
            
        except Exception as e:
            print(f"Display error: {e}")
            self.running = False
    
    def _update_display(self):
        """Update the display with current frame"""
        if self.running and self.root:
            with self.frame_lock:
                if self.current_frame is not None:
                    try:
                        # Convert BGR to RGB
                        rgb_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                        
                        # Convert to PIL Image
                        pil_image = Image.fromarray(rgb_frame)
                        
                        # Resize to fit canvas
                        canvas_width = self.canvas.winfo_width()
                        canvas_height = self.canvas.winfo_height()
                        if canvas_width > 1 and canvas_height > 1:
                            pil_image = pil_image.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
                        
                        # Convert to PhotoImage
                        self.photo = tk.PhotoImage(data=pil_image.tobytes(), format='PPM')
                        
                        # Update canvas
                        self.canvas.delete("all")
                        self.canvas.create_image(canvas_width//2, canvas_height//2, image=self.photo)
                        
                    except Exception as e:
                        print(f"Display update error: {e}")
            
            # Schedule next update
            if self.running:
                self.root.after(33, self._update_display)
    
    def update_frame(self, frame, status_text=""):
        """Update the frame to display"""
        with self.frame_lock:
            self.current_frame = frame.copy()
            if self.status_var and status_text:
                try:
                    self.status_var.set(status_text)
                except:
                    pass
    
    def update_status(self, status_text):
        """Update status text"""
        if self.status_var:
            try:
                self.status_var.set(status_text)
            except:
                pass
    
    def _quit_callback(self):
        """Handle quit button"""
        self.running = False
        if self.root:
            self.root.quit()
            self.root.destroy()
    
    def _reset_callback(self):
        """Handle reset button"""
        # This will be handled by the main application
        print("Reset requested via GUI")
    
    def _stats_callback(self):
        """Handle stats button"""
        # This will be handled by the main application
        print("Stats requested via GUI")
    
    def is_running(self):
        """Check if display is still running"""
        return self.running
    
    def close(self):
        """Close the display"""
        self.running = False
        if self.root:
            try:
                self.root.quit()
                self.root.destroy()
            except:
                pass

class CameraTamperingDetector:
    """Standalone camera tampering detector"""
    
    def __init__(self, 
                 img_width=300, 
                 img_height=300,
                 chroma_threshold=50.0,
                 gradient_threshold=30.0,
                 combined_threshold=40.0):
        
        # Processing parameters
        self.img_width = img_width
        self.img_height = img_height
        self.st_pool_size = 15  # Short-term pool size
        self.lt_pool_size = 30  # Long-term pool size
        self.lt_pool_cnt = 3    # Long-term pool count interval
        self.num_bins = 16
        
        # Thresholds
        self.chroma_threshold = chroma_threshold
        self.gradient_threshold = gradient_threshold
        self.combined_threshold = combined_threshold
        
        # Initialize queues
        self.st_frame_queue = HistogramQueue(self.st_pool_size)
        self.lt_frame_queue = HistogramQueue(self.lt_pool_size)
        self.chroma_hist_queue = HistogramQueue(self.lt_pool_size)
        self.gradient_hist_queue = HistogramQueue(self.lt_pool_size)
        self.cc_queue = HistogramQueue(self.st_pool_size)  # Chromaticity comparison
        self.gc_queue = HistogramQueue(self.st_pool_size)  # Gradient comparison
        
        # State flags
        self.cc_initialized = False
        self.gc_initialized = False
        self.counter = 0
        
        # Statistics
        self.stats = {
            "frames_processed": 0,
            "tampering_detections": 0,
            "chroma_violations": 0,
            "gradient_violations": 0,
            "combined_violations": 0,
            "average_chroma_diff": 0.0,
            "average_gradient_diff": 0.0,
            "start_time": time.time()
        }
        
        # Recent measurements for analysis
        self.recent_chroma_diffs = deque(maxlen=10)
        self.recent_gradient_diffs = deque(maxlen=10)
        
        # Alert state
        self.alert_active = False
        self.alert_start_time = None
        self.alert_duration = 5.0  # seconds
        self.alert_message = ""
        
        print(f"Tampering detector initialized:")
        print(f"  - Processing size: {img_width}x{img_height}")
        print(f"  - Chroma threshold: {chroma_threshold}")
        print(f"  - Gradient threshold: {gradient_threshold}")
        print(f"  - Combined threshold: {combined_threshold}")
    
    def resize_frame(self, frame):
        """Resize frame to standard processing size"""
        return cv2.resize(frame, (self.img_width, self.img_height))
    
    def calc_chroma_histogram(self, frame):
        """Calculate 2D chromaticity histogram (Hue-Saturation)"""
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Calculate 2D histogram for H-S channels
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        
        return hist
    
    def calc_gradient_histogram(self, frame):
        """Calculate gradient direction histogram"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Normalize
        normalized = gray.astype(np.float32) / 255.0
        
        # Calculate gradients
        grad_x = cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3)
        
        # Calculate magnitude and angle
        magnitude, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
        
        # Calculate histogram of gradient directions
        hist = cv2.calcHist([angle], [0], None, [self.num_bins], [0, 360])
        
        return hist
    
    def calculate_histogram_diff(self, hist1, hist2):
        """Calculate absolute difference between two histograms"""
        diff = cv2.absdiff(hist1, hist2)
        return np.sum(diff)
    
    def chromaticity_diff(self, current_frame):
        """Calculate chromaticity difference"""
        if not self.cc_initialized:
            print("Initializing chromaticity comparison...")
            
            # Initialize long-term pool histograms
            for frame in self.lt_frame_queue:
                chroma_hist = self.calc_chroma_histogram(frame)
                self.chroma_hist_queue.enqueue(chroma_hist)
            
            # Calculate initial short-term differences
            for frame in self.st_frame_queue:
                current_hist = self.calc_chroma_histogram(frame)
                diffs = []
                
                for lt_hist in self.chroma_hist_queue:
                    diff = self.calculate_histogram_diff(current_hist, lt_hist)
                    diffs.append(diff)
                
                avg_diff = np.mean(diffs) if diffs else 0.0
                self.cc_queue.enqueue(avg_diff)
            
            self.cc_initialized = True
            print("Chromaticity initialization complete")
        
        # Calculate difference for current frame
        current_hist = self.calc_chroma_histogram(current_frame)
        diffs = []
        
        for lt_hist in self.chroma_hist_queue:
            diff = self.calculate_histogram_diff(current_hist, lt_hist)
            diffs.append(diff)
        
        avg_diff = np.mean(diffs) if diffs else 0.0
        
        # Update queue
        if self.cc_queue.is_full():
            self.cc_queue.dequeue()
        self.cc_queue.enqueue(avg_diff)
        
        # Calculate overall average
        overall_avg = np.mean(list(self.cc_queue)) if len(self.cc_queue) > 0 else 0.0
        
        return overall_avg
    
    def gradient_direction_diff(self, current_frame):
        """Calculate gradient direction difference"""
        if not self.gc_initialized:
            print("Initializing gradient direction comparison...")
            
            # Initialize long-term pool histograms
            for frame in self.lt_frame_queue:
                grad_hist = self.calc_gradient_histogram(frame)
                self.gradient_hist_queue.enqueue(grad_hist)
            
            # Calculate initial short-term differences
            for frame in self.st_frame_queue:
                current_hist = self.calc_gradient_histogram(frame)
                diffs = []
                
                for lt_hist in self.gradient_hist_queue:
                    diff = self.calculate_histogram_diff(current_hist, lt_hist)
                    diffs.append(diff)
                
                avg_diff = np.mean(diffs) if diffs else 0.0
                self.gc_queue.enqueue(avg_diff)
            
            self.gc_initialized = True
            print("Gradient direction initialization complete")
        
        # Calculate difference for current frame
        current_hist = self.calc_gradient_histogram(current_frame)
        diffs = []
        
        for lt_hist in self.gradient_hist_queue:
            diff = self.calculate_histogram_diff(current_hist, lt_hist)
            diffs.append(diff)
        
        avg_diff = np.mean(diffs) if diffs else 0.0
        
        # Update queue
        if self.gc_queue.is_full():
            self.gc_queue.dequeue()
        self.gc_queue.enqueue(avg_diff)
        
        # Calculate overall average
        overall_avg = np.mean(list(self.gc_queue)) if len(self.gc_queue) > 0 else 0.0
        
        return overall_avg
    
    def detect_tampering(self, frame):
        """Main tampering detection logic"""
        # Resize frame for processing
        resized_frame = self.resize_frame(frame)
        
        # Add to short-term queue
        if self.st_frame_queue.is_full():
            self.st_frame_queue.dequeue()
        self.st_frame_queue.enqueue(resized_frame.copy())
        
        tampering_detected = False
        tampering_type = "none"
        chroma_diff = 0.0
        gradient_diff = 0.0
        
        # Process when both pools are filled
        if self.st_frame_queue.is_full() and self.lt_frame_queue.is_full():
            # Calculate differences
            chroma_diff = self.chromaticity_diff(resized_frame)
            gradient_diff = self.gradient_direction_diff(resized_frame)
            
            # Store recent measurements
            self.recent_chroma_diffs.append(chroma_diff)
            self.recent_gradient_diffs.append(gradient_diff)
            
            # Update statistics
            self.stats["average_chroma_diff"] = np.mean(list(self.recent_chroma_diffs))
            self.stats["average_gradient_diff"] = np.mean(list(self.recent_gradient_diffs))
            
            # Detect tampering based on thresholds
            chroma_violation = chroma_diff > self.chroma_threshold
            gradient_violation = gradient_diff > self.gradient_threshold
            combined_violation = (chroma_diff + gradient_diff) / 2 > self.combined_threshold
            
            if chroma_violation or gradient_violation or combined_violation:
                tampering_detected = True
                
                # Determine tampering type
                if chroma_violation and gradient_violation:
                    tampering_type = "severe_tampering"
                elif chroma_violation:
                    tampering_type = "color_obstruction"
                elif gradient_violation:
                    tampering_type = "physical_displacement"
                elif combined_violation:
                    tampering_type = "combined_tampering"
                
                # Update statistics
                self.stats["tampering_detections"] += 1
                if chroma_violation:
                    self.stats["chroma_violations"] += 1
                if gradient_violation:
                    self.stats["gradient_violations"] += 1
                if combined_violation:
                    self.stats["combined_violations"] += 1
                
                # Activate alert
                if not self.alert_active:
                    self.alert_active = True
                    self.alert_start_time = time.time()
                    self.alert_message = f"🚨 {tampering_type.replace('_', ' ').upper()} DETECTED! 🚨"
                    
                    print(f"\n{'='*60}")
                    print(f"TAMPERING ALERT: {tampering_type}")
                    print(f"Chroma diff: {chroma_diff:.2f} (threshold: {self.chroma_threshold})")
                    print(f"Gradient diff: {gradient_diff:.2f} (threshold: {self.gradient_threshold})")
                    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"{'='*60}\n")
        
        # Manage long-term pool
        if self.st_frame_queue.is_full():
            self.counter += 1
            
            if self.counter % self.lt_pool_cnt == 0:
                self.counter = 0
                
                # Move frame from short-term to long-term pool
                lt_frame = self.st_frame_queue.front().copy()
                
                if self.lt_frame_queue.is_full():
                    # Remove oldest frame and histograms
                    self.lt_frame_queue.dequeue()
                    if self.chroma_hist_queue.is_full():
                        self.chroma_hist_queue.dequeue()
                    if self.gradient_hist_queue.is_full():
                        self.gradient_hist_queue.dequeue()
                
                # Add new frame and calculate histograms
                self.lt_frame_queue.enqueue(lt_frame)
                
                chroma_hist = self.calc_chroma_histogram(lt_frame)
                self.chroma_hist_queue.enqueue(chroma_hist)
                
                grad_hist = self.calc_gradient_histogram(lt_frame)
                self.gradient_hist_queue.enqueue(grad_hist)
        
        # Update frame counter
        self.stats["frames_processed"] += 1
        
        return tampering_detected, tampering_type, chroma_diff, gradient_diff
    
    def draw_status_overlay(self, frame, chroma_diff, gradient_diff, tampering_detected, tampering_type):
        """Draw status overlay on frame"""
        current_time = time.time()
        
        # Create a copy to avoid modifying original
        display_frame = frame.copy()
        
        # Draw alert if active
        if self.alert_active:
            alert_age = current_time - self.alert_start_time
            if alert_age <= self.alert_duration:
                # Flash red background for alert
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], 100), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
                
                # Alert text
                cv2.putText(display_frame, self.alert_message, 
                           (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            else:
                self.alert_active = False
        
        # Status panel background
        panel_height = 120
        cv2.rectangle(display_frame, (0, display_frame.shape[0] - panel_height), 
                     (display_frame.shape[1], display_frame.shape[0]), (0, 0, 0), -1)
        
        # Status text
        y_offset = display_frame.shape[0] - panel_height + 20
        
        # Pool status
        pool_status = "Initializing..."
        if self.st_frame_queue.is_full() and self.lt_frame_queue.is_full():
            if self.cc_initialized and self.gc_initialized:
                pool_status = "✅ Active Monitoring"
            else:
                pool_status = "⚙️  Calibrating..."
        
        cv2.putText(display_frame, f"Status: {pool_status}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Measurements
        y_offset += 25
        cv2.putText(display_frame, f"Chroma: {chroma_diff:.1f}/{self.chroma_threshold} | Gradient: {gradient_diff:.1f}/{self.gradient_threshold}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Statistics
        y_offset += 20
        uptime = time.time() - self.stats["start_time"]
        fps = self.stats["frames_processed"] / max(1, uptime)
        cv2.putText(display_frame, f"Frames: {self.stats['frames_processed']} | FPS: {fps:.1f} | Detections: {self.stats['tampering_detections']}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Pool sizes
        y_offset += 20
        cv2.putText(display_frame, f"ST Pool: {len(self.st_frame_queue)}/{self.st_pool_size} | LT Pool: {len(self.lt_frame_queue)}/{self.lt_pool_size}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Threshold bars
        bar_width = 200
        bar_height = 8
        bar_x = display_frame.shape[1] - bar_width - 20
        bar_y = display_frame.shape[0] - 80
        
        # Chroma threshold bar
        cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
        chroma_fill = min(bar_width, int((chroma_diff / self.chroma_threshold) * bar_width))
        color = (0, 255, 0) if chroma_diff < self.chroma_threshold else (0, 0, 255)
        cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + chroma_fill, bar_y + bar_height), color, -1)
        cv2.putText(display_frame, f"Chroma", (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Gradient threshold bar
        bar_y += 15
        cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
        gradient_fill = min(bar_width, int((gradient_diff / self.gradient_threshold) * bar_width))
        color = (0, 255, 0) if gradient_diff < self.gradient_threshold else (0, 0, 255)
        cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + gradient_fill, bar_y + bar_height), color, -1)
        cv2.putText(display_frame, f"Gradient", (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return display_frame
    
    def get_stats(self):
        """Get current statistics"""
        uptime = time.time() - self.stats["start_time"]
        fps = self.stats["frames_processed"] / max(1, uptime)
        
        return {
            **self.stats,
            "uptime_seconds": uptime,
            "fps": fps,
            "pool_status": {
                "st_pool_size": len(self.st_frame_queue),
                "lt_pool_size": len(self.lt_frame_queue),
                "cc_initialized": self.cc_initialized,
                "gc_initialized": self.gc_initialized
            }
        }

def create_video_stream(source):
    """Create video stream from various sources"""
    print(f"Connecting to video source: {source}")
    
    # Try to open the source
    cap = cv2.VideoCapture(source)
    
    # Configure for RTSP streams
    if isinstance(source, str) and source.startswith('rtsp://'):
        print("Configuring for RTSP stream...")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size
        cap.set(cv2.CAP_PROP_FPS, 15)  # Set FPS
        
    # Check if opened successfully
    if not cap.isOpened():
        raise Exception(f"Failed to open video source: {source}")
    
    # Get stream properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video stream opened successfully:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    
    return cap

def save_detection_log(detector, output_dir="tampering_logs"):
    """Save detection statistics to file"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"tampering_log_{timestamp}.json")
    
    stats = detector.get_stats()
    stats["timestamp"] = datetime.now().isoformat()
    
    with open(log_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Detection log saved to: {log_file}")
    return log_file

def main():
    parser = argparse.ArgumentParser(description="Windows Compatible Camera Tampering Detection")
    
    # Video source options
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--rtsp", type=str, help="RTSP stream URL (e.g., rtsp://user:pass@ip:port/stream)")
    source_group.add_argument("--video", type=str, help="Video file path")
    source_group.add_argument("--webcam", type=int, help="Webcam device index (usually 0)")
    
    # Detection parameters
    parser.add_argument("--chroma-threshold", type=float, default=50.0, help="Chromaticity threshold (default: 50.0)")
    parser.add_argument("--gradient-threshold", type=float, default=30.0, help="Gradient threshold (default: 30.0)")
    parser.add_argument("--combined-threshold", type=float, default=40.0, help="Combined threshold (default: 40.0)")
    
    # Processing options
    parser.add_argument("--width", type=int, default=300, help="Processing width (default: 300)")
    parser.add_argument("--height", type=int, default=300, help="Processing height (default: 300)")
    parser.add_argument("--output", type=str, help="Save output video to file")
    parser.add_argument("--no-display", action="store_true", help="Don't display video window")
    parser.add_argument("--log-detections", action="store_true", help="Save detection log to file")
    parser.add_argument("--duration", type=int, help="Run for specified duration in seconds")
    parser.add_argument("--console-only", action="store_true", help="Console output only, no GUI")
    
    args = parser.parse_args()
    
    # Check if GUI libraries are available
    if not args.no_display and not args.console_only and not PIL_AVAILABLE:
        print("Warning: PIL/tkinter not available. Running in console-only mode.")
        args.console_only = True
    
    # Determine video source
    if args.rtsp:
        source = args.rtsp
    elif args.video:
        source = args.video
    elif args.webcam is not None:
        source = args.webcam
    
    try:
        # Create video stream
        cap = create_video_stream(source)
        
        # Initialize tampering detector
        detector = CameraTamperingDetector(
            img_width=args.width,
            img_height=args.height,
            chroma_threshold=args.chroma_threshold,
            gradient_threshold=args.gradient_threshold,
            combined_threshold=args.combined_threshold
        )
        
        # Initialize display manager if GUI is enabled
        display_manager = None
        if not args.no_display and not args.console_only:
            display_manager = WindowsDisplayManager()
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
            display_manager.start_display(width, height)

        # Initialize video writer if output specified
        video_writer = None
        if args.output:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = min(cap.get(cv2.CAP_PROP_FPS), 30.0)  # Limit to 30 FPS for output
            video_writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
            print(f"Saving output video to: {args.output}")
        
        # Initialize timing
        start_time = time.time()
        frame_count = 0
        last_stats_time = start_time
        last_console_update = start_time
        
        print("\nStarting tampering detection...")
        print("Press 'q' to quit, 's' to save stats, 'r' to reset detector")
        print("-" * 60)
        
        try:
            while True:
                # Check duration limit
                if args.duration and (time.time() - start_time) >= args.duration:
                    print(f"\nReached duration limit of {args.duration} seconds")
                    break
                
                # Read frame from video source
                ret, frame = cap.read()
                if not ret:
                    if isinstance(source, str) and not source.startswith('rtsp://'):
                        # End of video file
                        print("\nEnd of video file reached")
                        break
                    else:
                        # Network issue, try to reconnect
                        print("Connection lost, attempting to reconnect...")
                        cap.release()
                        time.sleep(1)
                        try:
                            cap = create_video_stream(source)
                            continue
                        except Exception as e:
                            print(f"Reconnection failed: {e}")
                            break
                
                frame_count += 1
                current_time = time.time()
                
                # Detect tampering
                tampering_detected, tampering_type, chroma_diff, gradient_diff = detector.detect_tampering(frame)
                
                # Create display frame with overlay
                display_frame = detector.draw_status_overlay(
                    frame, chroma_diff, gradient_diff, tampering_detected, tampering_type
                )
                
                # Update GUI display
                if display_manager and display_manager.is_running():
                    status_text = f"Frame {frame_count} | "
                    if tampering_detected:
                        status_text += f"🚨 {tampering_type.replace('_', ' ').upper()} DETECTED"
                    else:
                        status_text += f"✅ Normal Operation"
                    
                    display_manager.update_frame(display_frame, status_text)
                
                # Console output (less frequent to avoid spam)
                if args.console_only or (current_time - last_console_update) >= 2.0:
                    if args.console_only:
                        # Clear console and show current status
                        os.system('cls' if os.name == 'nt' else 'clear')
                        print("CAMERA TAMPERING DETECTION")
                        print("=" * 60)
                    
                    stats = detector.get_stats()
                    uptime = current_time - start_time
                    fps = frame_count / max(1, uptime)
                    
                    print(f"Frame: {frame_count:6d} | FPS: {fps:5.1f} | Uptime: {uptime:6.1f}s")
                    print(f"Chroma:   {chroma_diff:6.1f}/{detector.chroma_threshold:6.1f} | ", end="")
                    print(f"Gradient: {gradient_diff:6.1f}/{detector.gradient_threshold:6.1f}")
                    
                    pool_status = "Initializing"
                    if detector.st_frame_queue.is_full() and detector.lt_frame_queue.is_full():
                        if detector.cc_initialized and detector.gc_initialized:
                            pool_status = "Active"
                        else:
                            pool_status = "Calibrating"
                    
                    print(f"Status: {pool_status} | ST: {len(detector.st_frame_queue)}/{detector.st_pool_size} | LT: {len(detector.lt_frame_queue)}/{detector.lt_pool_size}")
                    
                    if tampering_detected:
                        print(f"🚨 TAMPERING DETECTED: {tampering_type.replace('_', ' ').upper()}")
                    
                    print(f"Detections: {stats['tampering_detections']} total | ", end="")
                    print(f"Chroma: {stats['chroma_violations']} | Gradient: {stats['gradient_violations']}")
                    
                    if not args.console_only:
                        print(f"Last update: {datetime.now().strftime('%H:%M:%S')}")
                    
                    print("-" * 60)
                    last_console_update = current_time
                
                # Write frame to output video
                if video_writer:
                    video_writer.write(display_frame)
                
                # Handle keyboard input for OpenCV display (if using cv2.imshow)
                if not args.no_display and args.console_only:
                    cv2.imshow('Camera Tampering Detection', display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q'):
                        print("\nQuitting...")
                        break
                    elif key == ord('s'):
                        if args.log_detections:
                            log_file = save_detection_log(detector)
                            print(f"Stats saved to: {log_file}")
                    elif key == ord('r'):
                        print("Resetting detector...")
                        # Reinitialize detector
                        detector = CameraTamperingDetector(
                            img_width=args.width,
                            img_height=args.height,
                            chroma_threshold=args.chroma_threshold,
                            gradient_threshold=args.gradient_threshold,
                            combined_threshold=args.combined_threshold
                        )
                
                # Check if GUI window was closed
                if display_manager and not display_manager.is_running():
                    print("\nGUI window closed")
                    break
                
                # Periodic stats output
                if (current_time - last_stats_time) >= 30.0:  # Every 30 seconds
                    stats = detector.get_stats()
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Periodic Stats:")
                    print(f"  Total frames: {stats['frames_processed']}")
                    print(f"  Tampering detections: {stats['tampering_detections']}")
                    print(f"  Average chroma diff: {stats['average_chroma_diff']:.2f}")
                    print(f"  Average gradient diff: {stats['average_gradient_diff']:.2f}")
                    print(f"  FPS: {stats['fps']:.2f}")
                    last_stats_time = current_time
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        except Exception as e:
            print(f"\nError during processing: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup
            print("\nCleaning up...")
            
            # Release video capture
            if cap:
                cap.release()
            
            # Release video writer
            if video_writer:
                video_writer.release()
                print(f"Output video saved to: {args.output}")
            
            # Close display
            if display_manager:
                display_manager.close()
            
            # Close any OpenCV windows
            cv2.destroyAllWindows()
            
            # Save final statistics
            if args.log_detections:
                try:
                    log_file = save_detection_log(detector)
                    print(f"Final detection log saved to: {log_file}")
                except Exception as e:
                    print(f"Failed to save log: {e}")
            
            # Print final summary
            print("\n" + "="*60)
            print("FINAL SUMMARY")
            print("="*60)
            
            stats = detector.get_stats()
            total_time = time.time() - start_time
            
            print(f"Total runtime: {total_time:.1f} seconds")
            print(f"Frames processed: {stats['frames_processed']}")
            print(f"Average FPS: {stats['fps']:.2f}")
            print(f"Tampering detections: {stats['tampering_detections']}")
            print(f"  - Chroma violations: {stats['chroma_violations']}")
            print(f"  - Gradient violations: {stats['gradient_violations']}")
            print(f"  - Combined violations: {stats['combined_violations']}")
            
            if stats['frames_processed'] > 0:
                detection_rate = (stats['tampering_detections'] / stats['frames_processed']) * 100
                print(f"Detection rate: {detection_rate:.2f}%")
            
            print(f"Final measurements:")
            print(f"  - Average chroma difference: {stats['average_chroma_diff']:.2f}")
            print(f"  - Average gradient difference: {stats['average_gradient_diff']:.2f}")
            
            print("="*60)
    
    except Exception as e:
        print(f"Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def create_test_scenarios():
    """Create synthetic test scenarios for tampering detection"""
    print("Creating synthetic test scenarios...")
    
    scenarios = []
    
    # Base frame - static scene
    base_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cv2.rectangle(base_frame, (100, 100), (300, 300), (0, 255, 0), -1)  # Green rectangle
    cv2.circle(base_frame, (400, 200), 50, (255, 0, 0), -1)  # Blue circle
    cv2.putText(base_frame, "Normal Scene", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    print("Generating normal operation frames...")
    # Scenario 1: Normal frames (50 frames)
    for i in range(50):
        frame = base_frame.copy()
        # Add slight noise to simulate real camera conditions
        noise = np.random.randint(-5, 5, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        scenarios.append(("normal", frame))
    
    print("Generating color obstruction scenario...")
    # Scenario 2: Color obstruction (hand covering lens)
    for i in range(25):
        frame = base_frame.copy()
        # Gradually darken and add red tint
        factor = 1.0 - (i / 25.0) * 0.9  # Darken by up to 90%
        frame = (frame * factor).astype(np.uint8)
        # Add red tint to simulate hand/object covering
        red_overlay = np.zeros_like(frame)
        red_overlay[:, :, 2] = 100  # Red channel
        frame = cv2.addWeighted(frame, 0.7, red_overlay, 0.3, 0)
        cv2.putText(frame, f"Obstruction {i+1}/25", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        scenarios.append(("color_obstruction", frame))
    
    print("Generating physical displacement scenario...")
    # Scenario 3: Physical displacement (camera moved)
    for i in range(25):
        frame = base_frame.copy()
        # Shift and rotate the frame content
        shift_x = i * 3
        shift_y = i * 2
        angle = i * 0.5
        
        # Translation and rotation matrix
        center = (frame.shape[1] // 2, frame.shape[0] // 2)
        M_rotate = cv2.getRotationMatrix2D(center, angle, 1.0)
        M_rotate[0, 2] += shift_x
        M_rotate[1, 2] += shift_y
        
        frame = cv2.warpAffine(frame, M_rotate, (640, 480))
        cv2.putText(frame, f"Displacement {i+1}/25", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        scenarios.append(("physical_displacement", frame))
    
    print("Generating lighting change scenario...")
    # Scenario 4: Dramatic lighting change
    for i in range(25):
        frame = base_frame.copy()
        if i < 12:
            # Sudden darkness
            brightness = -50 - i * 10
        else:
            # Sudden brightness
            brightness = 80 + (i - 12) * 15
        
        frame = np.clip(frame.astype(np.int16) + brightness, 0, 255).astype(np.uint8)
        cv2.putText(frame, f"Lighting {i+1}/25", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        scenarios.append(("lighting_change", frame))
    
    print("Generating severe tampering scenario...")
    # Scenario 5: Severe tampering (multiple effects)
    for i in range(20):
        frame = base_frame.copy()
        
        # Combine effects
        # Darkness
        frame = (frame * 0.3).astype(np.uint8)
        # Blur
        frame = cv2.GaussianBlur(frame, (15, 15), 0)
        # Color shift
        frame[:, :, 0] = np.clip(frame[:, :, 0] + 50, 0, 255)  # Increase blue
        # Noise
        noise = np.random.randint(-20, 20, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        cv2.putText(frame, f"SEVERE TAMPERING {i+1}/20", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        scenarios.append(("severe_tampering", frame))
    
    print(f"Created {len(scenarios)} test frames across 5 scenarios")
    return scenarios

def test_mode():
    """Run in test mode with synthetic scenarios"""
    print("Running Camera Tampering Detection in TEST MODE")
    print("="*60)
    
    # Create test scenarios
    scenarios = create_test_scenarios()
    
    # Initialize detector
    detector = CameraTamperingDetector(
        chroma_threshold=45.0,
        gradient_threshold=25.0,
        combined_threshold=35.0
    )
    
    # Initialize display if available
    display_manager = None
    if PIL_AVAILABLE:
        display_manager = WindowsDisplayManager()
        display_manager.start_display(640, 480)
    
    # Detection results tracking
    results = {
        "normal": {"total": 0, "detected": 0},
        "color_obstruction": {"total": 0, "detected": 0},
        "physical_displacement": {"total": 0, "detected": 0},
        "lighting_change": {"total": 0, "detected": 0},
        "severe_tampering": {"total": 0, "detected": 0}
    }
    
    print("\nProcessing test scenarios...")
    
    try:
        for i, (scenario_type, frame) in enumerate(scenarios):
            # Process frame
            tampering_detected, tampering_type, chroma_diff, gradient_diff = detector.detect_tampering(frame)
            
            # Track results
            results[scenario_type]["total"] += 1
            if tampering_detected:
                results[scenario_type]["detected"] += 1
            
            # Create display frame
            display_frame = detector.draw_status_overlay(
                frame, chroma_diff, gradient_diff, tampering_detected, tampering_type
            )
            
            # Update display
            if display_manager and display_manager.is_running():
                status = f"Test {i+1}/{len(scenarios)} | Scenario: {scenario_type} | "
                status += f"Detected: {tampering_type if tampering_detected else 'None'}"
                display_manager.update_frame(display_frame, status)
            else:
                # Console output
                print(f"Frame {i+1:3d}/{len(scenarios)} | {scenario_type:20s} | ", end="")
                if tampering_detected:
                    print(f"DETECTED: {tampering_type}")
                else:
                    print("Normal")
            
            # Small delay for visualization
            time.sleep(0.1)
            
            # Check if display was closed
            if display_manager and not display_manager.is_running():
                break
    
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    
    finally:
        # Cleanup
        if display_manager:
            display_manager.close()
        
        # Print results
        print("\n" + "="*60)
        print("TEST RESULTS SUMMARY")
        print("="*60)
        
        total_frames = sum(r["total"] for r in results.values())
        total_detections = sum(r["detected"] for r in results.values())
        
        print(f"Total frames processed: {total_frames}")
        print(f"Total detections: {total_detections}")
        print(f"Overall detection rate: {(total_detections/total_frames)*100:.1f}%")
        print()
        
        for scenario, data in results.items():
            if data["total"] > 0:
                detection_rate = (data["detected"] / data["total"]) * 100
                expected_rate = 0.0 if scenario == "normal" else 80.0  # Expected detection rate
                
                print(f"{scenario:20s}: {data['detected']:2d}/{data['total']:2d} ({detection_rate:5.1f}%) ", end="")
                
                if scenario == "normal":
                    # For normal frames, we want LOW detection rate
                    if detection_rate <= 10:
                        print("✅ GOOD (Low false positives)")
                    elif detection_rate <= 25:
                        print("⚠️  OK (Some false positives)")
                    else:
                        print("❌ POOR (Too many false positives)")
                else:
                    # For tampering scenarios, we want HIGH detection rate
                    if detection_rate >= 80:
                        print("✅ EXCELLENT")
                    elif detection_rate >= 60:
                        print("✅ GOOD")
                    elif detection_rate >= 40:
                        print("⚠️  OK")
                    else:
                        print("❌ POOR")
        
        # Final detector statistics
        stats = detector.get_stats()
        print(f"\nDetector Statistics:")
        print(f"  Frames processed: {stats['frames_processed']}")
        print(f"  Average chroma diff: {stats['average_chroma_diff']:.2f}")
        print(f"  Average gradient diff: {stats['average_gradient_diff']:.2f}")
        print(f"  Chroma violations: {stats['chroma_violations']}")
        print(f"  Gradient violations: {stats['gradient_violations']}")

if __name__ == "__main__":
    # Check if running in test mode
    if len(sys.argv) == 2 and sys.argv[1] == "test":
        test_mode()
    else:
        main()

        '''

