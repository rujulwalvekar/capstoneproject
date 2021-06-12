from dash_app.models import Patient


class Details(object):
    def __init__(self, request):
        self.request = request

    def get_patient_doctor_details(self):
        patients = Patient.objects.all().order_by('-last_interaction')
        user = self.request.user
        details = {'user': user, 'patients': patients}
        return details