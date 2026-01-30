from django.urls import path
from . import views  # Import your views

urlpatterns = [
    path('', views.index, name='index'),
    path('video_feed', views.video_feed, name='video_feed'),
    
    # --- THESE ARE CRITICAL FOR MOBILE & CHART ---
    path('mobile_process', views.process_mobile_frame, name='mobile_process'),
    path('stats', views.get_stats, name='get_stats'),  # <--- MUST MATCH fetch('/stats') in HTML
]