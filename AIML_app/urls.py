from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('video_feed', views.video_feed, name='video_feed'),
    path('mobile_process', views.process_mobile_frame, name='mobile_process'), # <--- NEW
    path('stats', views.get_stats, name='stats'),
]