from django.db import models

# Create your models here.
class Address(models.Model):
    address_id = models.AutoField(primary_key=True)
    street = models.CharField(max_length=255)
    city = models.CharField(max_length=100)
    postal_code = models.CharField(max_length=10)
    country = models.CharField(max_length=100)

class Guest(models.Model):
    guest_id = models.AutoField(primary_key=True)
    firstname = models.CharField(max_length=100)
    lastname = models.CharField(max_length=100)
    email = models.EmailField()
    phone = models.CharField(max_length=15)
    birth_date = models.DateField(null=True, blank=True)
    address = models.ForeignKey(Address, on_delete=models.CASCADE)
    guest_type = models.CharField(max_length=50)
    registration_date = models.DateField(auto_now_add=True)
    notes = models.TextField(null=True, blank=True)

class Employee(models.Model):
    employee_id = models.AutoField(primary_key=True)
    firstname = models.CharField(max_length=100)
    lastname = models.CharField(max_length=100)
    position = models.CharField(max_length=50)
    address = models.ForeignKey(Address, on_delete=models.CASCADE)

class RoomType(models.Model):
    room_type_id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=50)
    bed_count = models.IntegerField()
    price_per_night = models.DecimalField(max_digits=10, decimal_places=2)

class Room(models.Model):
    room_id = models.AutoField(primary_key=True)
    room_type = models.ForeignKey(RoomType, on_delete=models.CASCADE)
    room_number = models.CharField(max_length=10)
    is_occupied = models.BooleanField()

class Payment(models.Model):
    payment_id = models.AutoField(primary_key=True)
    total_accommodation = models.DecimalField(max_digits=10, decimal_places=2)
    total_expenses = models.DecimalField(max_digits=10, decimal_places=2)
    payment_date = models.DateField()
    is_paid = models.BooleanField()


class Reservation(models.Model):
    reservation_id = models.AutoField(primary_key=True)
    guest = models.ForeignKey(Guest, on_delete=models.CASCADE)
    room = models.ForeignKey(Room, on_delete=models.CASCADE)
    employee = models.ForeignKey(Employee, on_delete=models.CASCADE)
    creation_date = models.DateField()
    check_in_date = models.DateField()
    check_out_date = models.DateField(null=True, blank=True)
    payment = models.ForeignKey(Payment, on_delete=models.CASCADE)
    status = models.CharField(max_length=20)
