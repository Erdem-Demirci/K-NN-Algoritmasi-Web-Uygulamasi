from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_file, name='list'),
    path("arayuz", views.arayuz, name="arayuz"),
    path("secim", views.secim, name="secim")
    
]