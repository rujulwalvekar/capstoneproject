from datetime import datetime

from django.contrib.auth import login, authenticate
from django.contrib.auth.models import User
from django.http import HttpResponseRedirect
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
import os
from django.templatetags.static import static
from django.views.decorators.csrf import ensure_csrf_cookie

# Create your views here.

# display homepage
from Calendar import list_events
from .datetime_utilities import datetime_utilities
from .forms import RegistrationForm
from .models import Doctor, Patient
from .operations.PatientManager import PatientManager
from .operations.details import Details
from .operations.validate_events_information import ValidateEventsInformation


def get_homepage_user_details(request):
    details_obj = Details(request=request)
    details = details_obj.get_patient_doctor_details()
    email = details['user'].email
    username = details['user'].username
    patient_details = details['patients']
    return_data = {'username': username, 'patient_details': patient_details}

    try:
        print('email', request.user.email)
        events = list_events.get_events(str(email))
        # validate_events_informations = ValidateEventsInformation(events=events)
        # validated_events = validate_events_informations.update_info_of_events()
        return_data.update(
            {'calendar_events': events, 'day': datetime_utilities.day_of_week(datetime.today().weekday())})
    except Exception as e:
        print("Error while fetching calendar events", e)
    return return_data


def homepage(request):
    return render(request, "dashboard.html", get_homepage_user_details(request))


def results(request):
    # details_obj = Details(request=request)
    # details = details_obj.get_patient_doctor_details()

    # '''
    # The bottom code is temporary.
    # '''
    # patients = details.get('patients')
    # details['patient'] = patients[0]
    # ''''''
    return render(request, "results.html")


def profile(request, patient_id):
    # print('patient_it')
    # patient_id = 1
    # print('request_', request.value)
    details_obj = Details(request=request)
    details = details_obj.get_patient_doctor_details()

    '''
    The bottom code is temporary.
    '''
    patients = details.get('patients')
    details['patient'] = patients[patient_id - 1]
    ''''''
    print("PATIENT ", details)
    return render(request, "profile.html", details)


@csrf_exempt
def signin(request):
    print("Signing in user post")
    if request.method == 'POST':
        if request.POST['password1'] and request.POST['email']:
            try:
                user_email = request.POST['email']
                password = request.POST['password1']
                print(user_email, password)
                user = User.objects.filter(email=user_email).first()
                if user:
                    print('Logging in user: ', user)
                    login(request, user)
                    return HttpResponseRedirect('homepage/dashboard/')
                else:
                    return render(request, "sign-up.html")
            except Exception as e:
                print("Error while logging ing a user: {}".format(e))
    return render(request, "sign-in.html")


# @csrf_exempt
def signup(request):
    print('Signing up user GET')
    if request.method == 'POST':
        """
        Sign up a new user
        """
        if request.POST['password1'] and request.POST['email'] and request.POST['username']:
            user_name = request.POST['username']
            user_email = request.POST['email']
            password = request.POST['password1']
            try:
                user = User.objects.create_user(username=user_name, email=user_email, password=password)
                user.save()
                doctor = Doctor().create(name=user_name, email=user_email, phone_number=1234567899)
                doctor.save()
                login(request, user)
                return HttpResponseRedirect('homepage/dashboard/')
            except Exception as e:
                print("Error while signing up a user: {}".format(e))
    return render(request, "sign-up.html")



def addpatient(request):
    if request.method == 'POST':
        patient_manager_obj = PatientManager(patient_details=request.POST)
        required_patient_details = patient_manager_obj.all_required_fields_present()
        if required_patient_details:
            patient = Patient().create(
                **required_patient_details
            )
            patient.save()
            # return HttpResponseRedirect('homepage/dashboard/')
            return render(request, 'dashboard.html', get_homepage_user_details(request))
    return render(request, "addPatient.html")


def view_calendar_events(request):
    email = request.user.email
    events = list_events.get_events(str(email))



import sys
sys.path.insert(1, 'dl/model')

from backend_brain_pipeline import process_pipeline
from cet import cetpred
from pet import petpred
from xray import xraypred
from ecg import ecgpred
from breast import breastpred
from prostate import prostatepred
from glomerelu import glomerelupred

class OverwriteStorage(FileSystemStorage):

    def get_available_name(self, name, max_length=None):
        self.delete(name)
        return name

def mripredict(request):
    context={}
    paths = []
    print("*******************************")
    fs = OverwriteStorage()
    for count, x in enumerate(request.FILES.getlist("filelocation1")):
            def process(f):
                with open('/content/capstoneproject/Dashboard//media/file_' + str(count), 'wb+') as destination:
                    for chunk in f.chunks():
                        destination.write(chunk)
            process(x)
    # process_pipeline(paths, fname='/dash_app/static/dash_app/mriout.gif')
    context['a'] = 'The Results for MRI Scans are'
    context['b'] = 'Coloured regions indicate abnormality'
    context['c'] = 'static/assets/img/mriout.gif'
    context['d'] = 'Red - Necrotic and Non-enhancing tumor core (NCR) 10%'
    context['e'] = 'Green - Peritumoral Edema (ED) 8%'
    context['f'] = 'Blue - GD - Enhancing Tumor (ET) 12%'

    return render(request, 'results.html', context)

def cetpredict(request):

    context={}
    paths = []
    print("*******************************")
    fs = OverwriteStorage()
    for count, x in enumerate(request.FILES.getlist("filelocation2")):
            def process(f):
                with open('/content/capstoneproject/Dashboard//media/file_' + str(count), 'wb+') as destination:
                    for chunk in f.chunks():
                        destination.write(chunk)
            process(x)

    context['a'] = 'The Results for CET Scans are'
    context['b'] = 'Coloured regions indicate abnormality'
    context['c'] = 'static/assets/img/cetout.gif'
    context['d'] = 'Red - Necrotic and Non-enhancing tumor core (NCR) 10%'
    context['e'] = 'Green - Peritumoral Edema (ED) 8%'
    context['f'] = 'Blue - GD - Enhancing Tumor (ET) 12%'


    return render(request, 'results.html', context)

def petpredict(request):

    fs=FileSystemStorage()

    fileObj = request.FILES['filelocation5']
    filePathName5 = fs.save(fileObj.name, fileObj)
    filePathName5 = fs.url(fileObj.name)
    path = filePathName5
    print(filePathName5)

    print(path)

    a = petpred(path)
    context={}
    context['a'] = 'The Results for PET Scans are '
    if(a==0):
        context['b'] = 'Normal as per Ai'
    else:
        context['b'] = 'AbNormal as per Ai'

    context['c'] = 'static/assets/img/petout.gif'
    # path = '/content/capstoneproject/Dashboard/' + path
    # context['c'] = path
    return render(request, 'results.html', context)


def xraypredict(request):

    fs=FileSystemStorage()

    fileObj = request.FILES['filelocation6']
    filePathName6 = fs.save(fileObj.name, fileObj)
    filePathName6 = fs.url(fileObj.name)
    path = filePathName6
    print(filePathName6)

    print(path)

    # detect(image=path)
    a = xraypred(path)
    context={}
    context['a'] = 'The prediction for the XRay Image is '
    if(a==0):
      context['b'] = 'Normal Xray, no Pneumonia found by Ai'
      context['c'] = 'static/assets/img/xrayoutn.jpg'
    else:
      context['b'] = 'AbNormal Xray, Pneumonia found by Ai'
      context['c'] = 'static/assets/img/xrayouta.jpg'

    return render(request, 'results.html', context)


def ecgpredict(request):
    fs=FileSystemStorage()

    fileObj = request.FILES['filelocation7']
    filePathName7 = fs.save(fileObj.name, fileObj)
    filePathName7 = fs.url(fileObj.name)
    path = filePathName7
    print(filePathName7)

    print(path)
    a = ecgpred(path)
    context={}
    context['a'] = 'The Results for ECG are '
    if(a[0][0]==0):
      context['b'] = 'Non-ectopic Beats'
    elif(a[0][1]==0):
      context['b'] = 'Fusion Beats '
    elif(a[0][2]==0):
      context['b'] = 'AbNormal '
    elif(a[0][3]==0):
      context['b'] = 'AbNormal '
    elif(a[0][4]==0):
      context['b'] = 'AbNormal '
    context['c'] = 'static/assets/img/ecgout.jpg'
    return render(request, 'results.html', context)


def breastpredict(request):
    fs=FileSystemStorage()

    fileObj = request.FILES['filelocation8']
    filePathName8 = fs.save(fileObj.name, fileObj)
    filePathName8 = fs.url(fileObj.name)
    path = filePathName8
    print(filePathName8)

    print(path)
    a = breastpred(path)
    context={}
    context['a'] = 'The Results for Breast Cancer Cell detection are '
    if(a==0):
      context['b'] = 'Cancer cells are present'
    else:
      context['b'] = 'Cancer Cells are not present '

    context['c'] = 'static/assets/img/breastout.gif'

    return render(request, 'results.html', context)


def glomerelupredict(request):

    fs=FileSystemStorage()
    print("Inside prostate function")
    fileObj = request.FILES['filelocation9']
    filePathName9 = fs.save(fileObj.name, fileObj)
    filePathName9 = fs.url(fileObj.name)
    path = filePathName9
    print(filePathName9)

    print(path)
    a = glomerelupred(path)
    context={}
    context['a'] = 'The Results for Glomerelu'
    context['b']= 'The Coloured spots indicate Anamoly'
    context['c'] = 'static/assets/img/glomereluout.png'

    return render(request, 'results.html', context)

def prostatepredict(request):

    fs=FileSystemStorage()
    print("Inside prostate function")
    fileObj = request.FILES['filelocation10']
    filePathName10 = fs.save(fileObj.name, fileObj)
    filePathName10 = fs.url(fileObj.name)
    path = filePathName10
    print(filePathName10)

    print(path)
    a = prostatepred(path)
    context={}
    context['a'] = 'The Results for Prostate Cancer is '
    context['b']=  'Tissue Level is ' + str(a)
    context['c'] = 'static/assets/img/prostateout.png'

    return render(request, 'results.html', context)
