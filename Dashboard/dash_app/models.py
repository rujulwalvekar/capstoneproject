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
    email = models.EmailField(max_length=100)
    phone_number = models.IntegerField(max_length=10)
    address = models.CharField(max_length=500)
    blood_type = models.CharField(max_length=5)
    hospital_name = models.CharField(max_length=50)
    visit_type = models.CharField(max_length=50)
    last_interaction = models.DateField(default=datetime.now())
