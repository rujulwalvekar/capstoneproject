from datetime import datetime

from django.contrib.auth import login, authenticate
from django.contrib.auth.models import User
from django.http import HttpResponseRedirect
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt

# Create your views here.

# display homepage
from Calendar import list_events
from .datetime_utilities import datetime_utilities
from .forms import RegistrationForm
from .models import Doctor, Patient
from .operations.PatientManager import PatientManager
from .operations.details import Details
from .operations.validate_events_information import ValidateEventsInformation


def homepage(request):
    details_obj = Details(request=request)
    details = details_obj.get_patient_doctor_details()
    email = details['user'].email
    username = details['user'].username
    patient_details = details['patients']
    return_data = {'username': username, 'patient_details': patient_details}

    try:
        print('email', request.user.email)
        events = list_events.get_events(str(email))
        validate_events_informations = ValidateEventsInformation(events=events)
        validated_events = validate_events_informations.update_info_of_events()
        return_data.update({'calendar_events': validated_events, 'day': datetime_utilities.day_of_week(datetime.today().weekday())})
    except Exception as e:
        print("Error while fetching calendar events", e)
    return render(request, "dashboard.html", return_data)


def results(request):
    details_obj = Details(request=request)
    details = details_obj.get_patient_doctor_details()

    '''
    The bottom code is temporary.
    '''
    patients = details.get('patients')
    details['patient'] = patients[0]
    ''''''
    return render(request, "results.html", details)


def profile(request):
    details_obj = Details(request=request)
    details = details_obj.get_patient_doctor_details()

    '''
    The bottom code is temporary.
    '''
    patients = details.get('patients')
    details['patient'] = patients[0]
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

# def signup(request):
#     print('Signing up user GET')
#     context = {}
#     if request.method == 'POST':
#         """
#         Sign up a new user
#         """
#         form = RegistrationForm(request.POST)
#         if not form.is_valid():
#             # form.save()
#             print('valid')
#             if form.cleaned_data['password1'] and form.cleaned_data['email'] and form.cleaned_data['username']:
#                 print('got ')
#                 try:
#                     user_name = form.cleaned_data['username']
#                     user_email = form.cleaned_data['email']
#                     password1 = form.cleaned_data['password1']
#                     password2 = form.cleaned_data['password2']
#                     # account = authenticate(email=user_email, password=password1)
#                     # user = User.objects.create_user(username=user_name, email=user_email, password=password)
#                     # user.save()
#                     doctor = Doctor().create(name=user_name, email=user_email, phone_number=1234567899)
#                     doctor.save()
#                     login(request, account)
#                     return HttpResponseRedirect('homepage/dashboard/')
#                 except Exception as e:
#                     print("Error while signing up a user: {}".format(e))
#         else:
#             context['form'] = form
#     else:
#         form = RegistrationForm()
#         context['form'] = form
#
#     return render(request, "sign-up.html", context)

def addpatient(request):
    if request.method == 'POST':
        patient_manager_obj = PatientManager(patient_details=request.POST)
        required_patient_details = patient_manager_obj.all_required_fields_present()
        if required_patient_details:
            patient = Patient().create(
                *required_patient_details
            )
            patient.save()
            # return HttpResponseRedirect('homepage/dashboard/')
    return render(request, "addPatient.html")


def view_calendar_events(request):
    email = request.user.email
    events = list_events.get_events(str(email))
