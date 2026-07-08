from django.urls import path
from . import views

from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [

    #home 
    path('', views.home, name='home'),

    #events management
    path('event/<int:event_id>/', views.event, name='event'),
    path('edit/<int:event_id>/', views.edit_event, name='edit_event'),
    path('delete/<int:event_id>/', views.delete_event, name='delete_event'),
    
    # Face Recognition page
    path('event/<int:event_id>/face-recognition/', views.face_recognition_page, name='face_recognition_page'),


    # Camera RTSP URLs
    path('camera/add/<int:event_id>/', views.add_rtsp_camera, name='add_camera'),
    path('camera/delete/<int:camera_id>/', views.delete_rtsp_camera, name='delete_camera'),
    path('camera/pin/<int:camera_id>/', views.pin_camera, name='pin_camera'),
    path('camera/unpin/<int:camera_id>/', views.unpin_camera, name='unpin_camera'),
    path('camera/list/<int:event_id>/', views.get_event_cameras, name='get_cameras'),
    path('camera/feed/<int:camera_id>/', views.video_feed, name='video_feed'),
    path('camera/test/', views.test_rtsp_connection, name='test_rtsp'),

    # Face Recognition URLs
    path('face/upload-model/<int:event_id>/', views.upload_face_model, name='upload_face_model'),
    path('face/add-to-database/<int:event_id>/', views.add_face_to_database, name='add_face_to_database'),
    path('face/models/<int:event_id>/', views.get_face_models, name='get_face_models'),
    path('face/database/<int:event_id>/', views.get_face_database, name='get_face_database'),
    path('face/delete-model/<int:model_id>/', views.delete_face_model, name='delete_face_model'),
    path('face/delete-from-database/<int:face_id>/', views.delete_face_from_database, name='delete_face_from_database'),
    path('face/activate-model/<int:model_id>/', views.activate_face_model, name='activate_face_model'),
    path('face/detect-image/<int:event_id>/', views.detect_faces_in_image, name='detect_faces_in_image'),

    # Object detection feature removed

    # Model Confidence URL
    path('model/update-confidence/<int:model_id>/', views.update_model_confidence, name='update_model_confidence'),
    # API to toggle face recognition for an event (updates running streams immediately)
    path('api/face-recognition/toggle/<int:event_id>/', views.toggle_face_recognition, name='toggle_face_recognition'),
    # API to get face recognition status for an event
    path('api/face-recognition/status/<int:event_id>/', views.get_face_recognition_status, name='get_face_recognition_status'),
    
    # Detection Logs URLs
    path('logs/<int:event_id>/', views.get_detection_logs, name='get_detection_logs'),
    path('logs/delete/<int:log_id>/', views.delete_detection_log, name='delete_detection_log'),
    path('logs/clear/<int:event_id>/', views.clear_detection_logs, name='clear_detection_logs'),
    path('logs/report/<int:event_id>/', views.generate_detection_report, name='generate_detection_report'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)