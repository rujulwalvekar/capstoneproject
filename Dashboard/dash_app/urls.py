from django.urls import path

from . import views

urlpatterns = [
    path('', views.homepage, name='dash_app'),
    # path('rujul', views.rujul, name='rujul'),
]