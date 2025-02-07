from django.contrib import admin
from .models import Address, Guest, Employee, RoomType, Room, Payment, Reservation

admin.site.register(Address)
admin.site.register(Guest)
admin.site.register(Employee)
admin.site.register(RoomType)
admin.site.register(Room)
admin.site.register(Payment)
admin.site.register(Reservation)
