from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path("get_disciplines/", views.get_disciplines, name="get_disciplines"),
    path("confirm_discipline/", views.confirm_discipline, name="confirm_discipline"),
]
