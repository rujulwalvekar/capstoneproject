from django.shortcuts import render


# Create your views here.

# display homepage
def homepage(request):
    return render(request, "dashboard.html")

def results(request):
    return render(request, "results.html")

def profile(request):
    return render(request, "profile.html")