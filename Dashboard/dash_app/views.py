from django.shortcuts import render
from django.shortcuts import render, redirect
# Create your views here.

from django.http import HttpResponse
# display homepage
def homepage(request):
    return render(request, "dashboard.html")