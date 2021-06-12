from django.contrib.auth import login
from django.contrib.auth.models import User
from django.http import HttpResponseRedirect
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt

# Create your views here.

# display homepage
from Calendar import list_events
from .models import Doctor, Patient
from .operations.PatientManager import PatientManager
from .operations.details import Details


def homepage(request):
    details_obj = Details(request=request)
    details = details_obj.get_patient_doctor_details()
    email = details['user'].email
    username = details['user'].username
    patient_details = details['patients']
    events = list_events.get_events(str(email))
    return_data = {'username': username, 'patient_details': patient_details, 'calendar_events': events}
    print('events', events)
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
    print("Signing in user get")
    if request.method == 'POST':
        if request.POST['password_input_type'] and request.POST['email_input_type']:
            try:
                user_email = request.POST['email_input_type']
                password = request.POST['password_input_type']
                print(user_email, password)
                user = User.objects.filter(email=user_email, password=password)
                if user:
                    print('Logging in user: ', user)
                    login(request, user)
                    return HttpResponseRedirect('homepage/dashboard/')
            except Exception as e:
                print("Error while logging ing a user: {}".format(e))
    return render(request, "sign-in.html")


@csrf_exempt
def signup(request):
    print('Signing up user GET')
    if request.method == 'POST':
        """
        Sign up a new user
        """
        if request.POST['password_input_type'] and request.POST['email_input_type'] and request.POST['name_input_type']:
            user_name = request.POST['name_input_type']
            user_email = request.POST['email_input_type']
            password = request.POST['password_input_type']
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
                *required_patient_details
            )
            patient.save()
            # return HttpResponseRedirect('homepage/dashboard/')
    return render(request, "addPatient.html")


def view_calendar_events(request):
    email = request.user.email
    events = list_events.get_events(str(email))
