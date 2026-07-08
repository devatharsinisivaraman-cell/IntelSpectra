from django.shortcuts import render, redirect, get_object_or_404
from sih_app.models import Event, RtspTable, FaceRecognitionModel, FaceDatabase, DetectionLog, AIModelEnabled
from django.http import JsonResponse, StreamingHttpResponse
import cv2
import threading
from datetime import datetime
from django.core.files.base import ContentFile
import io
import numpy as np
import json
import face_recognition
import os
from django.conf import settings


#HOME
def home(request):
    if request.method == 'POST':
        try:
            title = request.POST.get('title')
            description = request.POST.get('description')
            location = request.POST.get('location')
            event_type = request.POST.get('event_type')
            
            print("Received form data:", title, description, location, event_type) 
            
            event = Event.objects.create(
                event_name=title,
                event_description=description,
                event_location=location,
                event_type=event_type
            )
            
            print("Event created:", event.id)  
            return redirect('home')
            
        except Exception as e:
            print("Error:", str(e))  
            return render(request, 'home.html', {
                'events': Event.objects.all().order_by('-event_created_date'),
                'error': str(e)
            })
    
    events = Event.objects.all().order_by('-event_created_date')
    return render(request, 'home.html', {'events': events})

def edit_event(request, event_id):
    if request.method == 'POST':
        try:
            event = get_object_or_404(Event, id=event_id)
            event.event_name = request.POST.get('title')
            event.event_description = request.POST.get('description')
            event.event_location = request.POST.get('location')
            event.event_type = request.POST.get('event_type')
            event.save()
            print("Event updated:", event.id)
            return redirect('home')
        except Exception as e:
            print("Error updating:", str(e))
            return redirect('home')
    
    event = get_object_or_404(Event, id=event_id)
    return render(request, 'home.html', {
        'events': Event.objects.all().order_by('-event_created_date'),
        'edit_event': event
    })

def delete_event(request, event_id):
    if request.method == 'POST':
        try:
            event = get_object_or_404(Event, id=event_id)
            event.delete()
            print("Event deleted:", event_id)
        except Exception as e:
            print("Error deleting:", str(e))
    return redirect('home')

# EVENT PAGE
def event(request, event_id):
    event = get_object_or_404(Event, id=event_id)
    cameras = RtspTable.objects.filter(event_name=event)
    pinned_cameras = cameras.filter(is_pinned=True)[:2]
    normal_cameras = cameras.filter(is_pinned=False)  
    
    return render(request, 'event.html', {
        'event': event,
        'pinned_cameras': pinned_cameras,
        'normal_cameras': normal_cameras,
        'total_cameras': cameras.count()
    })


def face_recognition_page(request, event_id):
    """Face Recognition management page"""
    event = get_object_or_404(Event, id=event_id)
    return render(request, 'face_recognition.html', {'event': event})


#RTSP CONNECTION MANAGEMENT

camera_streams = {}

# Cache face encodings per event_id to avoid reloading from disk every frame
face_encodings_cache = {}

def get_face_encodings_for_event(event_id):
    """Load face encodings from FaceDatabase for the event and cache them.

    Returns list of tuples: (person_name, encoding)
    """
    try:
        if event_id in face_encodings_cache:
            return face_encodings_cache[event_id]

        enc_list = []
        faces = FaceDatabase.objects.filter(event_name_id=event_id)
        for face in faces:
            try:
                # Build absolute path to the stored image
                img_path = os.path.join(settings.MEDIA_ROOT, face.face_image.name)
                if not os.path.exists(img_path):
                    continue
                img = face_recognition.load_image_file(img_path)
                encs = face_recognition.face_encodings(img)
                if len(encs) > 0:
                    enc_list.append((face.person_name, encs[0]))
            except Exception:
                continue

        face_encodings_cache[event_id] = enc_list
        return enc_list
    except Exception:
        return []

def add_rtsp_camera(request, event_id):
    """Add a new camera/stream to an event - supports multiple protocols"""
    if request.method == 'POST':
        try:
            event = get_object_or_404(Event, id=event_id)
            camera_name = request.POST.get('camera_name')
            rtsp_url = request.POST.get('rtsp_url')
            stream_type = request.POST.get('stream_type', 'rtsp')
            camera_location = request.POST.get('camera_location', '')
            
            # Validate and process URL based on stream type
            if stream_type == 'usb':
                # For USB cameras, expect device index (0, 1, 2, etc.)
                try:
                    device_index = int(rtsp_url)
                    rtsp_url = str(device_index)
                except ValueError:
                    rtsp_url = '0'  # Default to first USB camera
            
            # Create camera entry
            rtsp_camera = RtspTable.objects.create(
                event_name=event,
                camera_name=camera_name,
                rtsp_url=rtsp_url,
                stream_type=stream_type
            )
            
            print(f"RTSP Camera added: {camera_name} - {rtsp_url}")
            
            return JsonResponse({
                'status': 'success',
                'camera_id': rtsp_camera.id,
                'message': 'Camera added successfully'
            })
            
        except Exception as e:
            print(f"Error adding camera: {str(e)}")
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=400)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)

def delete_rtsp_camera(request, camera_id):
    """Delete an RTSP camera"""
    if request.method == 'POST':
        try:
            camera = get_object_or_404(RtspTable, id=camera_id)
            camera.delete()
            
            # Stop stream if active and cleanup properly
            if camera_id in camera_streams:
                try:
                    camera_streams[camera_id].stop()
                except Exception:
                    pass
                del camera_streams[camera_id]
            
            return JsonResponse({
                'status': 'success',
                'message': 'Camera deleted successfully'
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=400)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)

def get_event_cameras(request, event_id):
    """Get all cameras for an event"""
    try:
        event = get_object_or_404(Event, id=event_id)
        cameras = RtspTable.objects.filter(event_name=event).values(
            'id', 'camera_name', 'rtsp_url', 'stream_type', 'added_at'
        )
        
        return JsonResponse({
            'status': 'success',
            'cameras': list(cameras)
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=400)

def test_rtsp_connection(request):
    """Test stream connection before adding - supports all protocols"""
    if request.method == 'POST':
        try:
            rtsp_url = request.POST.get('rtsp_url')
            stream_type = request.POST.get('stream_type', 'rtsp')
            
            # Handle USB camera
            if stream_type == 'usb':
                try:
                    device_index = int(rtsp_url)
                except ValueError:
                    device_index = 0
                cap = cv2.VideoCapture(device_index)
            else:
                # Try to connect to network stream or file
                cap = cv2.VideoCapture(rtsp_url)
            
            success = cap.isOpened()
            
            if success:
                # Read one frame to verify
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    return JsonResponse({
                        'status': 'success',
                        'message': f'{stream_type.upper()} connection successful'
                    })
            
            cap.release()
            return JsonResponse({
                'status': 'error',
                'message': f'Failed to connect to {stream_type.upper()} stream'
            }, status=400)
            
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=400)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)

def pin_camera(request, camera_id):
    """Pin a camera to the pinned screens"""
    if request.method == 'POST':
        try:
            camera = get_object_or_404(RtspTable, id=camera_id)
            
            # Check if already 2 pinned cameras for this event
            pinned_count = RtspTable.objects.filter(
                event_name=camera.event_name, 
                is_pinned=True
            ).count()
            
            if pinned_count >= 2:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Maximum 2 cameras can be pinned'
                }, status=400)
            
            camera.is_pinned = True
            camera.save()
            
            return JsonResponse({
                'status': 'success',
                'message': 'Camera pinned successfully'
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=400)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)

def unpin_camera(request, camera_id):
    """Unpin a camera from pinned screens"""
    if request.method == 'POST':
        try:
            camera = get_object_or_404(RtspTable, id=camera_id)
            camera.is_pinned = False
            camera.save()
            
            return JsonResponse({
                'status': 'success',
                'message': 'Camera unpinned successfully'
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=400)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)

class VideoCamera:
    """Class to handle video streaming from multiple sources with AI detection"""
    def __init__(self, rtsp_url, camera_id, event_id, enable_detection=True, stream_type='rtsp'):
        self.rtsp_url = rtsp_url
        self.camera_id = camera_id
        self.event_id = event_id
        self.stream_type = stream_type
        self.enable_detection = enable_detection
        self.video = None
        self.lock = threading.Lock()
        self.stopped = False
        self.frame_count = 0
        self.detection_manager = None
        self.last_detections = {}  # Store last detections to draw on all frames
        self.last_face_annotations = []  # Store last face annotations (list of ((top,right,bottom,left), name))
        self.face_recognition_mode = False  # when True, only show face bounding boxes and skip other detection overlays
        self._ai_flags_last_checked = 0
        self.latest_frame = None
        self.stopped = False
        
        # Face tracking variables for smooth stable recognition
        self.tracked_faces = {}  # {track_id: {'bbox': (t,r,b,l), 'name': str, 'confidence': float, 'frames_tracked': int}}
        self.next_track_id = 0
        self.max_tracking_distance = 80  # Maximum pixel distance to consider same face
        self.min_confidence_frames = 3  # Require 3 consistent frames before showing name
        self.max_lost_frames = 10  # Remove track after this many frames without detection
        
        self._recognition_thread = threading.Thread(target=self._recognition_worker, daemon=True)
        self.connect()
        # Start recognition worker
        try:
            self._recognition_thread.start()
        except Exception:
            pass

    def refresh_ai_flags(self):
        """Refresh AI flags from DB for this event. Cheap check cached per seconds/frames."""
        try:
            cfg = AIModelEnabled.objects.filter(target_event_id=self.event_id).first()
            if cfg:
                self.face_recognition_mode = bool(cfg.is_face_recognition_enabled)
            else:
                self.face_recognition_mode = False
        except Exception:
            self.face_recognition_mode = False
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union for two bounding boxes"""
        # box format: (top, right, bottom, left)
        t1, r1, b1, l1 = box1
        t2, r2, b2, l2 = box2
        
        # Calculate intersection
        x_left = max(l1, l2)
        y_top = max(t1, t2)
        x_right = min(r1, r2)
        y_bottom = min(b1, b2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (r1 - l1) * (b1 - t1)
        box2_area = (r2 - l2) * (b2 - t2)
        
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return iou
    
    def calculate_distance(self, box1, box2):
        """Calculate center distance between two bounding boxes"""
        # box format: (top, right, bottom, left)
        t1, r1, b1, l1 = box1
        t2, r2, b2, l2 = box2
        
        # Calculate centers
        c1_x = (l1 + r1) / 2
        c1_y = (t1 + b1) / 2
        c2_x = (l2 + r2) / 2
        c2_y = (t2 + b2) / 2
        
        # Euclidean distance
        distance = ((c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2) ** 0.5
        return distance
    
    def update_tracked_faces(self, new_detections):
        """Update face tracking with new detections
        
        new_detections: list of ((top, right, bottom, left), name) tuples
        Returns: list of ((top, right, bottom, left), name, track_id) with stable tracking
        """
        current_time = self.frame_count
        
        # Mark all tracks as not updated
        for track_id in self.tracked_faces:
            self.tracked_faces[track_id]['updated'] = False
        
        matched_detections = []
        
        # Match new detections to existing tracks
        for bbox, name in new_detections:
            best_match_id = None
            best_match_score = 0
            
            # Find best matching track based on IOU and distance
            for track_id, track_info in self.tracked_faces.items():
                iou = self.calculate_iou(bbox, track_info['bbox'])
                distance = self.calculate_distance(bbox, track_info['bbox'])
                
                # Use IOU for matching (more robust than distance)
                if iou > 0.3 and distance < self.max_tracking_distance:
                    score = iou
                    if score > best_match_score:
                        best_match_score = score
                        best_match_id = track_id
            
            if best_match_id is not None:
                # Update existing track
                track = self.tracked_faces[best_match_id]
                track['bbox'] = bbox
                track['last_seen'] = current_time
                track['frames_tracked'] += 1
                track['updated'] = True
                
                # Update name with voting mechanism for stability
                if name == track['name']:
                    track['name_confidence'] = min(track['name_confidence'] + 1, 10)
                else:
                    track['name_confidence'] -= 1
                    if track['name_confidence'] <= 0:
                        track['name'] = name
                        track['name_confidence'] = 1
                
                # Only show name if tracked for minimum frames
                display_name = track['name'] if track['frames_tracked'] >= self.min_confidence_frames else "Detecting..."
                matched_detections.append((bbox, display_name, best_match_id))
            else:
                # Create new track
                track_id = self.next_track_id
                self.next_track_id += 1
                
                self.tracked_faces[track_id] = {
                    'bbox': bbox,
                    'name': name,
                    'name_confidence': 1,
                    'frames_tracked': 1,
                    'last_seen': current_time,
                    'updated': True
                }
                
                # Show "Detecting..." for new faces
                display_name = "Detecting..." if self.min_confidence_frames > 1 else name
                matched_detections.append((bbox, display_name, track_id))
        
        # Remove tracks that haven't been updated for too long
        tracks_to_remove = []
        for track_id, track_info in self.tracked_faces.items():
            if not track_info.get('updated', False):
                if current_time - track_info['last_seen'] > self.max_lost_frames:
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracked_faces[track_id]
        
        return matched_detections
    
    def connect(self):
        """Connect to video stream - supports multiple protocols"""
        try:
            # Handle different stream types
            if self.stream_type == 'usb':
                # USB camera - convert to integer device index
                try:
                    device_index = int(self.rtsp_url)
                except ValueError:
                    device_index = 0
                self.video = cv2.VideoCapture(device_index)
                
            elif self.stream_type == 'file':
                # Video file path
                self.video = cv2.VideoCapture(self.rtsp_url)
                
            elif self.stream_type in ['rtsp', 'rtmp', 'http', 'hls', 'ip', 'other']:
                # Network streams - OpenCV handles these automatically
                # For better performance with network streams
                self.video = cv2.VideoCapture(self.rtsp_url)
                
                # Set buffer size for network streams to reduce latency
                if self.stream_type in ['rtsp', 'rtmp']:
                    self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
            else:
                # Default fallback
                self.video = cv2.VideoCapture(self.rtsp_url)
            
            # Verify connection
            if not self.video.isOpened():
                self.video = None
                print(f"Failed to open stream: {self.rtsp_url} (type: {self.stream_type})")
                
        except Exception as e:
            print(f"Error connecting to stream: {e}")
            self.video = None
    
    def get_frame(self):
        """Get a frame from the video stream with optional AI detection"""
        if self.video is None or not self.video.isOpened():
            self.connect()
            if self.video is None:
                return None
        
        with self.lock:
            success, frame = self.video.read()
            if not success:
                self.connect()
                return None

            # store latest frame for background recognition worker (only if needed)
            try:
                if self.face_recognition_mode:
                    self.latest_frame = frame.copy()
                else:
                    self.latest_frame = None
            except Exception:
                self.latest_frame = None

            self.frame_count += 1
            
            # Run detection every 5 frames to save processing power
            if self.enable_detection and self.detection_manager:
                if self.frame_count % 5 == 0:
                    try:
                        # Run detection and update last detections
                        self.last_detections = self.detection_manager.detect_frame(frame)
                        
                        # Log detections every 30 frames (reduce database writes)
                        if self.last_detections and self.frame_count % 30 == 0:
                            self.log_detections(self.last_detections, frame)
                    except Exception as e:
                        pass
                
                # Draw the last detected bounding boxes on EVERY frame for smooth display
                if self.last_detections:
                    try:
                        frame = self.detection_manager.draw_detections(frame, self.last_detections)
                    except Exception as e:
                        pass

            # Face recognition: run every 5 frames and cache annotations for smooth display
            try:
                # refresh AI flags occasionally (every 60 frames)
                if self.frame_count - self._ai_flags_last_checked > 60:
                    self.refresh_ai_flags()
                    self._ai_flags_last_checked = self.frame_count

                if self.frame_count % 5 == 0:
                    try:
                        # If face recognition is not enabled for this event, skip processing
                        if not self.face_recognition_mode:
                            # ensure annotations cleared
                            self.last_face_annotations = []
                            self.tracked_faces.clear()  # Clear tracking when disabled
                            raise Exception("face recognition disabled")

                        # Convert to RGB for face_recognition library
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Detect face locations
                        locs = face_recognition.face_locations(rgb, model='hog')  # Use HOG for faster detection

                        raw_detections = []

                        if len(locs) > 0:
                            # Get face encodings and match with known faces in database
                            encs = face_recognition.face_encodings(rgb, locs)
                            known = get_face_encodings_for_event(self.event_id)
                            known_names = [k[0] for k in known]
                            known_vecs = [k[1] for k in known]

                            for (top, right, bottom, left), enc in zip(locs, encs):
                                name = "Unknown"  # Default label for unrecognized faces
                                if len(known_vecs) > 0:
                                    matches = face_recognition.compare_faces(known_vecs, enc, tolerance=0.25)
                                    if True in matches:
                                        idx = matches.index(True)
                                        name = known_names[idx]
                                raw_detections.append(((top, right, bottom, left), name))

                        # Apply tracking for smooth and stable recognition
                        self.last_face_annotations = self.update_tracked_faces(raw_detections)
                        
                        # Explicitly delete large objects to free memory
                        del rgb
                        if 'encs' in locals():
                            del encs
                    except Exception:
                        self.last_face_annotations = []                # Draw last face annotations on every frame, but only when face recognition is enabled
                # Only draw bounding boxes for RECOGNIZED faces (skip "Unknown" and "Detecting...")
                if self.face_recognition_mode and self.last_face_annotations:
                    try:
                        for item in self.last_face_annotations:
                            # Unpack based on tuple length (with or without track_id)
                            if len(item) == 3:
                                (top, right, bottom, left), name, track_id = item
                            else:
                                (top, right, bottom, left), name = item
                                track_id = None
                            
                            # Skip drawing if face is "Unknown" or "Detecting..."
                            if name == "Detecting..." or name == "Unknown" or not name:
                                continue
                            
                            # Log recognized face detection (every 30 frames to avoid spam)
                            if self.frame_count % 30 == 0:
                                try:
                                    self._save_detection_log(frame, name, (top, right, bottom, left))
                                except Exception:
                                    pass
                            
                            # Only draw for recognized faces (with actual names)
                            box_color = (0, 255, 0)  # Green for recognized
                            text_bg_color = (0, 255, 0)
                            
                            # Draw bounding box with thicker line for better visibility
                            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
                            
                            # Display the person's name label
                            label = name
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.6
                            thickness = 2
                            
                            # Get text size for background rectangle
                            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                            
                            # Draw filled rectangle as background for text
                            cv2.rectangle(frame, (left, top - text_height - 10), 
                                        (left + text_width + 6, top), text_bg_color, -1)
                            
                            # Draw text label in black color on colored background
                            cv2.putText(frame, label, (left + 3, top - 5), 
                                      font, font_scale, (0, 0, 0), thickness)
                            
                            # Optional: Draw track ID for debugging (small corner indicator)
                            if track_id is not None and False:  # Set to True to enable
                                cv2.putText(frame, f"#{track_id}", (right - 30, top + 15),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)
                    except Exception as e:
                        pass
            except Exception:
                pass
            
            # Encode frame as JPEG
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                return jpeg.tobytes()
            return None
    
    def stop(self):
        """Stop the video camera and cleanup resources"""
        self.stopped = True
        try:
            # Wait for recognition thread to finish
            if hasattr(self, '_recognition_thread') and self._recognition_thread.is_alive():
                self._recognition_thread.join(timeout=1.0)
        except Exception:
            pass
        
        try:
            if self.video is not None:
                self.video.release()
                self.video = None
        except Exception:
            pass
        
        # Clear frame references to free memory
        try:
            self.latest_frame = None
            self.last_face_annotations = []
            self.tracked_faces.clear()
        except Exception:
            pass
    
    def __del__(self):
        """Clean up resources"""
        try:
            self.stop()
        except Exception:
            pass
    
    def _save_detection_log(self, frame, detected_name, bbox):
        """Save face detection to database with snapshot"""
        try:
            # Extract face region from frame
            top, right, bottom, left = bbox
            
            # Add some padding
            padding = 20
            top = max(0, top - padding)
            left = max(0, left - padding)
            bottom = min(frame.shape[0], bottom + padding)
            right = min(frame.shape[1], right + padding)
            
            # Crop face region
            face_img = frame[top:bottom, left:right]
            
            # Encode as JPEG
            ret, jpeg = cv2.imencode('.jpg', face_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ret:
                return
            
            # Create ContentFile for Django
            img_io = io.BytesIO(jpeg.tobytes())
            img_file = ContentFile(img_io.getvalue(), name=f'face_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}.jpg')
            
            # Save to database
            DetectionLog.objects.create(
                event_name_id=self.event_id,
                camera_id=self.camera_id,
                detection_type='face',
                detected_label=detected_name,
                confidence_score=0.95,  # You can calculate actual confidence if available
                bounding_box={'top': int(top), 'right': int(right), 'bottom': int(bottom), 'left': int(left)},
                frame_snapshot=img_file
            )
        except Exception as e:
            print(f"Error saving detection log: {e}")

    def _recognition_worker(self):
        """Background worker that performs face recognition on the latest frame.

        It downscales frames for performance and updates `self.last_face_annotations`.
        """
        try:
            while not self.stopped:
                try:
                    # Check if stopped
                    if self.stopped:
                        break
                    
                    # refresh flags periodically
                    self.refresh_ai_flags()

                    frame = None
                    try:
                        with self.lock:
                            if self.latest_frame is not None:
                                frame = self.latest_frame.copy()
                    except Exception:
                        frame = None

                    if frame is None:
                        # nothing to process yet
                        threading.Event().wait(0.05)
                        continue

                    # Downscale for performance
                    h, w = frame.shape[:2]
                    target_w = 320
                    scale = 1.0
                    if w > target_w:
                        scale = target_w / float(w)
                        small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                    else:
                        small = frame

                    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

                    # Detect face locations on small image
                    locs_small = face_recognition.face_locations(rgb_small)

                    annotations = []
                    if self.face_recognition_mode:
                        # Only bounding boxes (no name matching)
                        for (top, right, bottom, left) in locs_small:
                            # map back to original coords
                            top_o = int(top / scale)
                            right_o = int(right / scale)
                            bottom_o = int(bottom / scale)
                            left_o = int(left / scale)
                            annotations.append(((top_o, right_o, bottom_o, left_o), None))
                    else:
                        # compute encodings on small image and match
                        encs_small = face_recognition.face_encodings(rgb_small, locs_small)
                        known = get_face_encodings_for_event(self.event_id)
                        known_names = [k[0] for k in known]
                        known_vecs = [k[1] for k in known]

                        for (top, right, bottom, left), enc in zip(locs_small, encs_small):
                            name = None
                            if len(known_vecs) > 0:
                                matches = face_recognition.compare_faces(known_vecs, enc, tolerance=0.45)
                                if True in matches:
                                    idx = matches.index(True)
                                    name = known_names[idx]

                            top_o = int(top / scale)
                            right_o = int(right / scale)
                            bottom_o = int(bottom / scale)
                            left_o = int(left / scale)
                            annotations.append(((top_o, right_o, bottom_o, left_o), name))

                    # update shared annotations
                    try:
                        with self.lock:
                            self.last_face_annotations = annotations
                    except Exception:
                        pass

                except Exception:
                    # resilience: don't break worker on errors
                    pass

                # small sleep to control CPU
                try:
                    threading.Event().wait(0.08)
                except Exception:
                    pass
        except Exception:
            # Final catch-all for thread safety
            pass

def run_onnx_inference(frame):
    input_blob = preprocess(frame)
    outputs = onnx_session.run([onnx_output], {onnx_input: input_blob})[0]
    return outputs

def draw_boxes(frame, preds):
    # preds = [x1, y1, x2, y2, score, class_id] per row
    for det in preds:
        x1, y1, x2, y2, score, cls = det
        if score < 0.50:
            continue

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, str(cls), (x1, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return frame


def model_processing(frame):

    # Convert JPEG bytes → image if needed
    np_frame = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)

    preds = run_onnx_inference(np_frame)
    
    # draw boxes
    processed = draw_boxes(np_frame, preds)

    # Encode back to JPEG
    ret, jpeg = cv2.imencode(".jpg", processed)
    return jpeg.tobytes()


def gen_frames(camera_id, enable_detection=True):
    """Generate frames for streaming - supports multiple protocols"""
    if camera_id not in camera_streams:
        try:
            camera = RtspTable.objects.get(id=camera_id)
            video_camera = VideoCamera(
                camera.rtsp_url, 
                camera_id, 
                camera.event_name_id,
                enable_detection=enable_detection,
                stream_type=camera.stream_type
            )
            # Set initial face recognition mode based on AIModelEnabled
            cfg = AIModelEnabled.objects.filter(target_event_id=camera.event_name_id).first()
            if cfg:
                video_camera.face_recognition_mode = cfg.is_face_recognition_enabled
            camera_streams[camera_id] = video_camera
        except RtspTable.DoesNotExist:
            return
    
    camera = camera_streams[camera_id]
    
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue
        # model_processing handled separately; frame already annotated by VideoCamera
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def video_feed(request, camera_id):
    """Stream video feed from RTSP camera"""
    try:
        camera = get_object_or_404(RtspTable, id=camera_id)
        
        return StreamingHttpResponse(
            gen_frames(camera_id),
            content_type='multipart/x-mixed-replace; boundary=frame'
        )
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=400)


def toggle_face_recognition(request, event_id):
    """API endpoint to toggle face recognition for an event.

    Expects POST with 'enabled'='true'|'false'. Updates AIModelEnabled and pushes to running streams.
    """
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)

    try:
        enabled = request.POST.get('enabled', 'false').lower() == 'true'

        # Ensure AIModelEnabled row exists for event
        cfg, _ = AIModelEnabled.objects.get_or_create(target_event_id=event_id)
        cfg.is_face_recognition_enabled = enabled
        cfg.save()

        # Push update to running camera streams for this event
        for cam_id, cam in camera_streams.items():
            try:
                if cam.event_id == event_id:
                    cam.face_recognition_mode = enabled
                    print(f"[toggle_face_recognition] Updated cam {cam_id} for event {event_id} -> face_recognition_mode={enabled}")
            except Exception:
                continue

        return JsonResponse({'status': 'success', 'enabled': enabled})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)


def get_face_recognition_status(request, event_id):
    """API endpoint to get current face recognition status for an event."""
    try:
        cfg = AIModelEnabled.objects.filter(target_event_id=event_id).first()
        if cfg:
            enabled = cfg.is_face_recognition_enabled
        else:
            enabled = False
        
        return JsonResponse({'status': 'success', 'enabled': enabled})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)


# ========================================
# FACE RECOGNITION VIEWS
# ========================================

def upload_face_model(request, event_id):
    """Upload a face recognition model"""
    if request.method == 'POST':
        try:
            event = get_object_or_404(Event, id=event_id)
            model_name = request.POST.get('model_name')
            model_file = request.FILES.get('model_file')
            
            if not model_file:
                return JsonResponse({
                    'status': 'error',
                    'message': 'No model file provided'
                }, status=400)
            
            # Create face recognition model
            face_model = FaceRecognitionModel.objects.create(
                event_name=event,
                model_name=model_name,
                model_file=model_file,
                is_active=False
            )
            
            return JsonResponse({
                'status': 'success',
                'model_id': face_model.id,
                'message': 'Face recognition model uploaded successfully'
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=400)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)


def add_face_to_database(request, event_id):
    """Add a face to the recognition database"""
    if request.method == 'POST':
        try:
            event = get_object_or_404(Event, id=event_id)
            person_name = request.POST.get('person_name')
            face_image = request.FILES.get('face_image')
            
            if not face_image or not person_name:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Person name and face image required'
                }, status=400)
            
            # Create face database entry
            face_db = FaceDatabase.objects.create(
                event_name=event,
                person_name=person_name,
                face_image=face_image
            )

            # Invalidate cache for this event so new face is picked up
            try:
                face_encodings_cache.pop(event.id, None)
            except Exception:
                pass
            
            return JsonResponse({
                'status': 'success',
                'face_id': face_db.id,
                'message': f'Face for {person_name} added successfully'
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=400)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)


def get_face_models(request, event_id):
    """Get all face recognition models for an event"""
    try:
        event = get_object_or_404(Event, id=event_id)
        models = FaceRecognitionModel.objects.filter(event_name=event).values(
            'id', 'model_name', 'is_active', 'confidence_threshold', 'created_at'
        )
        
        return JsonResponse({
            'status': 'success',
            'models': list(models)
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=400)


def get_face_database(request, event_id):
    """Get all registered faces for an event"""
    try:
        event = get_object_or_404(Event, id=event_id)
        faces = FaceDatabase.objects.filter(event_name=event)
        
        faces_data = []
        for face in faces:
            faces_data.append({
                'id': face.id,
                'person_name': face.person_name,
                'image_url': face.face_image.url if face.face_image else '',
                'created_at': face.created_at.isoformat()
            })
        
        return JsonResponse({
            'status': 'success',
            'faces': faces_data
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=400)


def delete_face_model(request, model_id):
    """Delete a face recognition model"""
    if request.method == 'POST':
        try:
            model = get_object_or_404(FaceRecognitionModel, id=model_id)
            model.delete()
            
            return JsonResponse({
                'status': 'success',
                'message': 'Face model deleted successfully'
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=400)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)


def delete_face_from_database(request, face_id):
    """Delete a face from the database"""
    if request.method == 'POST':
        try:
            face = get_object_or_404(FaceDatabase, id=face_id)
            face.delete()
            # Invalidate cache for this event to remove deleted face
            try:
                face_encodings_cache.pop(face.event_name_id, None)
            except Exception:
                pass
            return JsonResponse({
                'status': 'success',
                'message': 'Face deleted from database'
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=400)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)


def activate_face_model(request, model_id):
    """Activate/Deactivate a face recognition model"""
    if request.method == 'POST':
        try:
            model = get_object_or_404(FaceRecognitionModel, id=model_id)
            is_active = request.POST.get('is_active', 'false').lower() == 'true'
            
            model.is_active = is_active
            model.save()
            
            return JsonResponse({
                'status': 'success',
                'message': f'Face model {model.model_name} {"activated" if is_active else "deactivated"}'
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=400)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)


def detect_faces_in_image(request, event_id):
    """Detect faces in an uploaded image"""
    if request.method == 'POST':
        try:
            event = get_object_or_404(Event, id=event_id)
            image_file = request.FILES.get('image')
            
            if not image_file:
                return JsonResponse({
                    'status': 'error',
                    'message': 'No image provided'
                }, status=400)
            
            # Read image
            nparr = np.frombuffer(image_file.read(), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Simple face detection using Haar Cascade
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            detections = []
            for (x, y, w, h) in faces:
                detections.append({
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h),
                    'confidence': 0.95
                })
            
            return JsonResponse({
                'status': 'success',
                'detections': detections,
                'count': len(detections)
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=400)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)


# Object detection feature removed: related views deleted


def update_model_confidence(request, model_id):
    """Update confidence threshold for a face recognition model"""
    if request.method == 'POST':
        try:
            face_model = FaceRecognitionModel.objects.filter(id=model_id).first()
            confidence = float(request.POST.get('confidence', 0.5))

            if not face_model:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Face model not found'
                }, status=400)

            face_model.confidence_threshold = confidence
            face_model.save()

            return JsonResponse({
                'status': 'success',
                'message': 'Confidence threshold updated'
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=400)

    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)


# ========================================
# DETECTION LOGS VIEWS
# ========================================

def get_detection_logs(request, event_id):
    """API endpoint to get detection logs for an event"""
    try:
        page = int(request.GET.get('page', 1))
        page_size = int(request.GET.get('page_size', 20))
        detection_type = request.GET.get('type', None)
        
        # Query logs
        logs = DetectionLog.objects.filter(event_name_id=event_id)
        
        if detection_type:
            logs = logs.filter(detection_type=detection_type)
        
        # Pagination
        start = (page - 1) * page_size
        end = start + page_size
        total_count = logs.count()
        logs = logs[start:end]
        
        # Serialize logs
        logs_data = []
        for log in logs:
            logs_data.append({
                'id': log.id,
                'camera_id': log.camera_id,
                'detection_type': log.detection_type,
                'detected_label': log.detected_label,
                'confidence_score': log.confidence_score,
                'bounding_box': log.bounding_box,
                'frame_snapshot': log.frame_snapshot.url if log.frame_snapshot else None,
                'created_at': log.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                'time_ago': get_time_ago(log.created_at)
            })
        
        return JsonResponse({
            'status': 'success',
            'logs': logs_data,
            'total': total_count,
            'page': page,
            'page_size': page_size,
            'has_next': end < total_count
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=400)


def delete_detection_log(request, log_id):
    """Delete a specific detection log"""
    if request.method == 'POST':
        try:
            log = DetectionLog.objects.get(id=log_id)
            
            # Delete the image file if exists
            if log.frame_snapshot:
                log.frame_snapshot.delete()
            
            log.delete()
            
            return JsonResponse({
                'status': 'success',
                'message': 'Log deleted successfully'
            })
        except DetectionLog.DoesNotExist:
            return JsonResponse({
                'status': 'error',
                'message': 'Log not found'
            }, status=404)
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=400)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)


def clear_detection_logs(request, event_id):
    """Clear all detection logs for an event"""
    if request.method == 'POST':
        try:
            logs = DetectionLog.objects.filter(event_name_id=event_id)
            count = logs.count()
            
            # Delete all images
            for log in logs:
                if log.frame_snapshot:
                    log.frame_snapshot.delete()
            
            logs.delete()
            
            return JsonResponse({
                'status': 'success',
                'message': f'Deleted {count} logs'
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=400)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)


def generate_detection_report(request, event_id):
    """Generate a detection report for an event"""
    try:
        from django.db.models import Count
        from datetime import timedelta
        
        event = get_object_or_404(Event, id=event_id)
        
        # Get date range
        date_from = request.GET.get('date_from', None)
        date_to = request.GET.get('date_to', None)
        
        logs = DetectionLog.objects.filter(event_name_id=event_id)
        
        if date_from:
            logs = logs.filter(created_at__gte=date_from)
        if date_to:
            logs = logs.filter(created_at__lte=date_to)
        
        # Statistics
        total_detections = logs.count()
        
        # Group by detection type
        by_type = logs.values('detection_type').annotate(count=Count('id'))
        
        # Group by detected label
        by_label = logs.values('detected_label').annotate(count=Count('id')).order_by('-count')[:10]
        
        # Group by camera
        by_camera = logs.values('camera_id').annotate(count=Count('id'))
        
        # Recent detections
        recent = logs.order_by('-created_at')[:5]
        recent_data = []
        for log in recent:
            recent_data.append({
                'detected_label': log.detected_label,
                'detection_type': log.detection_type,
                'created_at': log.created_at.strftime('%Y-%m-%d %H:%M:%S')
            })
        
        return JsonResponse({
            'status': 'success',
            'report': {
                'event_name': event.event_name,
                'event_location': event.event_location,
                'total_detections': total_detections,
                'by_type': list(by_type),
                'by_label': list(by_label),
                'by_camera': list(by_camera),
                'recent_detections': recent_data
            }
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=400)


def get_time_ago(dt):
    """Helper function to get human-readable time ago"""
    from datetime import datetime, timezone
    
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    diff = now - dt
    
    seconds = diff.total_seconds()
    
    if seconds < 60:
        return f"{int(seconds)} seconds ago"
    elif seconds < 3600:
        return f"{int(seconds / 60)} minutes ago"
    elif seconds < 86400:
        return f"{int(seconds / 3600)} hours ago"
    else:
        return f"{int(seconds / 86400)} days ago"