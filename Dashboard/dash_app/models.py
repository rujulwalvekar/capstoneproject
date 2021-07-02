from datetime import datetime

from django.db import models


# Create your models here.
class Doctor(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField(max_length=100)
    phone_number = models.IntegerField()

    @classmethod
    def create(cls, name, email, phone_number):
        doctor = cls(
            name=name,
            email=email,
            phone_number=phone_number
        )
        return doctor


class Patient(models.Model):
    name = models.CharField(max_length=100)
    gender = models.CharField(max_length=100, default='Male')
    date_of_birth = models.CharField(max_length=100, default=datetime.now())
    email = models.EmailField(max_length=100)
    phone_number = models.CharField(max_length=10)
    birth_place = models.CharField(max_length=100, default='Pune')
    country = models.CharField(max_length=100, default='India')
    state = models.CharField(max_length=100, default='Maharashtra')

    address = models.CharField(max_length=500, default="Default address")
    allergies = models.CharField(max_length=500, default="Dust Allergy")
    medications = models.CharField(max_length=500, default="Cetrizine")
    observations = models.CharField(max_length=500, default="Calcium Deficit")
    blood_type = models.CharField(max_length=5, default='b +ve')
    hospital_name = models.CharField(max_length=50, default='DMH')
    visit_type = models.CharField(max_length=50, default='Consult')
    last_interaction = models.DateField(default=datetime.now())

    @classmethod
    def create(cls, name, gender, date_of_birth,address,allergies,medications,observations, email, phone_number, hospital_name, birth_place, country, state):
        patient = cls(
            name=name,
            gender=gender,
            date_of_birth=date_of_birth,
            address=address,
            allergies=allergies,
            medications=medications,
            observations=observations,
            email=email,
            phone_number=phone_number,
            hospital_name=hospital_name,
            birth_place=birth_place,
            country=country,
            state=state
        )
        return patient