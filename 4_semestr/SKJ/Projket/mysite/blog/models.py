from django.db import models
from datetime import timedelta
from django import forms
from django.contrib.auth.models import User

# Create your models here.
class Address(models.Model):
    address_id = models.AutoField(primary_key=True)
    street = models.CharField(max_length=255)
    city = models.CharField(max_length=100)
    postal_code = models.CharField(max_length=10)
    country = models.CharField(max_length=100)

    def __str__(self):
        return f"{self.street}, {self.city}, {self.postal_code}"

class Guest(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, blank=True, null=True)
    guest_id = models.AutoField(primary_key=True)
    firstname = models.CharField(max_length=100)
    lastname = models.CharField(max_length=100)
    email = models.EmailField(max_length=100)
    phone = models.CharField(max_length=30, blank=True, null=True)
    birth_date = models.DateField()
    street = models.CharField(max_length=255)
    city = models.CharField(max_length=100)
    postal_code = models.CharField(max_length=10)
    country = models.CharField(max_length=100)
    guest_type = models.CharField(max_length=50)
    registration_date = models.DateField(auto_now_add=True)
    notes = models.TextField(blank=True, null=True)

    def __str__(self):
        return self.firstname + " " + self.lastname

class Employee(models.Model):
    employee_id = models.AutoField(primary_key=True)
    firstname = models.CharField(max_length=100)
    lastname = models.CharField(max_length=100)
    position = models.CharField(max_length=50)
    street = models.CharField(max_length=255)
    city = models.CharField(max_length=100)
    postal_code = models.CharField(max_length=10)
    country = models.CharField(max_length=100)

class RoomType(models.Model):
    room_type_id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100)
    bed_count = models.IntegerField()
    price_per_night = models.DecimalField(max_digits=10, decimal_places=2, default=1000)  # Přidáno pole pro cenu
    description = models.TextField(blank=True, null=True)  # Přidáno pole pro popis

    def __str__(self):
        return f"{self.name} ({self.bed_count} lůžek) - {self.price_per_night} Kč/noc"

class Room(models.Model):
    room_id = models.AutoField(primary_key=True)
    room_type = models.ForeignKey(RoomType, on_delete=models.CASCADE)
    room_number = models.CharField(max_length=20, unique=True)
    is_occupied = models.BooleanField(default=False)

    def __str__(self):
        return f"Pokoje {self.room_number} ({self.room_type.name})"


class Payment(models.Model):
    payment_id = models.AutoField(primary_key=True)
    total_accommodation = models.DecimalField(max_digits=10, decimal_places=2)
    total_expenses = models.DecimalField(max_digits=10, decimal_places=2)
    payment_date = models.DateField(blank=True, null=True)
    is_paid = models.BooleanField(default=False)

class Reservation(models.Model):
    reservation_id = models.AutoField(primary_key=True)
    guest = models.ForeignKey(Guest, on_delete=models.CASCADE)
    room = models.ForeignKey(Room, on_delete=models.CASCADE)
    creation_date = models.DateField(auto_now_add=True)
    check_in_date = models.DateField()
    check_out_date = models.DateField()
    payment = models.ForeignKey(Payment, on_delete=models.CASCADE)
    status = models.CharField(max_length=30)
    accommodation_price = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)

class Service(models.Model):
    service_id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True, null=True)
    price = models.DecimalField(max_digits=10, decimal_places=2, default=0)  # <-- přidej toto pole

    def __str__(self):
        return f"{self.name} ({self.price} Kč)"

class ServiceUsage(models.Model):
    usage_id = models.AutoField(primary_key=True)
    reservation = models.ForeignKey(Reservation, on_delete=models.CASCADE)
    service = models.ForeignKey(Service, on_delete=models.CASCADE)
    quantity = models.IntegerField()
    total_price = models.DecimalField(max_digits=10, decimal_places=2)

class Feedback(models.Model):
    feedback_id = models.AutoField(primary_key=True)
    guest = models.ForeignKey(Guest, on_delete=models.CASCADE)
    reservation = models.ForeignKey(Reservation, on_delete=models.CASCADE)
    rating = models.IntegerField()
    comment = models.TextField(blank=True, null=True)
    feedback_date = models.DateField(auto_now_add=True)

class EmployeeForm(forms.ModelForm):
    class Meta:
        model = Employee
        fields = ['firstname', 'lastname', 'position', 'street', 'city', 'postal_code', 'country']