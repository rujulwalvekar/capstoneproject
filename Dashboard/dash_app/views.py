from django.shortcuts import render


# Create your views here.

# display homepage
def homepage(request):
    return render(request, "dashboard.html")
