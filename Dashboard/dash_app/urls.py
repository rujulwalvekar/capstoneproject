from django.urls import path

from . import views

urlpatterns = [
    path('', views.homepage, name='dashboard'),
    path('result', views.results, name='result'),
    path('overview', views.profile, name='overview'),
    path('sign-in', views.signin, name='sign-in'),
    path('sign-up', views.signup, name='sign-up'),
    path('new', views.addpatient, name='addPatient'),
    # path('rujul', views.rujul, name='rujul'),
]