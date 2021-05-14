from django.shortcuts import render


# Create your views here.

# display homepage
def homepage(request):
    return render(request, "dashboard.html")

def results(request):
    return render(request, "results.html")

def profile(request):
    return render(request, "profile.html")

def signin(request):
    return render(request, "sign-in.html")

def signup(request):
    return render(request, "sign-up.html")

def addpatient(request):
    return render(request, "addPatient.html")