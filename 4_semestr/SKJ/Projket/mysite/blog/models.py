from django.db import models
from datetime import timedelta

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

    def __str__(self):
        return self.name

class Room(models.Model):
    room_id = models.AutoField(primary_key=True)
    room_type = models.ForeignKey(RoomType, on_delete=models.CASCADE, null=False, blank=False)
    room_number = models.CharField(max_length=10)
    is_occupied = models.BooleanField()

    def __str__(self):
        return f"{self.room_id} (Room {self.room_number})"
    

class Payment(models.Model):
    payment_id = models.AutoField(primary_key=True)
    total_accommodation = models.DecimalField(max_digits=10, decimal_places=2)
    total_expenses = models.DecimalField(max_digits=10, decimal_places=2)
    payment_date = models.DateField(null=True, blank=True)
    is_paid = models.BooleanField()


class Reservation(models.Model):
    reservation_id = models.AutoField(primary_key=True)
    guest = models.ForeignKey(Guest, on_delete=models.CASCADE)
    room = models.ForeignKey(Room, on_delete=models.CASCADE)
    creation_date = models.DateField(auto_now_add=True)
    check_in_date = models.DateField()
    check_out_date = models.DateField(null=True, blank=True)
    payment = models.OneToOneField(Payment, on_delete=models.CASCADE, null=True, blank=True)  # Jedna rezervace má jedno platby
    status = models.CharField(max_length=20)

    def save(self, *args, **kwargs):
        # Vytvoření nebo aktualizace platby
        if not self.payment:
            # Výpočet ceny za ubytování
            if self.check_in_date and self.check_out_date:
                nights = (self.check_out_date - self.check_in_date).days
                price_per_night = self.room.room_type.price_per_night
                total_accommodation = nights * price_per_night
            else:
                total_accommodation = 0

            # Vytvoření nové platby
            payment = Payment.objects.create(
                total_accommodation=total_accommodation,
                total_expenses=0,  # Výchozí hodnota pro další výdaje
                payment_date=None,  # Datum platby bude nastaveno později
                is_paid=False
            )
            self.payment = payment
        else:
            # Aktualizace existující platby
            if self.check_in_date and self.check_out_date:
                nights = (self.check_out_date - self.check_in_date).days
                price_per_night = self.room.room_type.price_per_night
                self.payment.total_accommodation = nights * price_per_night
                self.payment.save()

        super().save(*args, **kwargs)

class Positions(models.Model):
    position_id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=50)
    description = models.TextField(null=True, blank=True)
    salary = models.DecimalField(max_digits=10, decimal_places=2)