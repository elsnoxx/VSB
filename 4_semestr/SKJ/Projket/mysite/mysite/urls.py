from django.urls import path
from django.contrib import admin
from blog import views

urlpatterns = [
    path('', views.index, name='index'),
    path('admin/', admin.site.urls),
    path('guests/', views.guest_list, name='guest_list'),
    path('guests/<int:guest_id>/', views.guest_detail, name='guest_detail'),
    path('guests/new/', views.guest_create, name='guest_create'),
    path('guests/<int:guest_id>/edit/', views.guest_update, name='guest_update'),
    path('guests/<int:guest_id>/delete/', views.guest_delete, name='guest_delete'),
    path('reservations/', views.reservation_list, name='reservation_list'),
    path('reservations/<int:reservation_id>/', views.reservation_detail, name='reservation_detail'),
    path('add-guest/', views.add_guest, name='add_guest'),
    path('reservation/create/<int:room_id>/', views.create_reservation, name='create_reservation'),
    path('room/management/', views.room_management, name='room_management'),
    path('room/create/', views.room_create, name='room_create'),
    path('room/<int:room_id>/update/', views.room_update, name='room_update'),
    path('room/<int:room_id>/delete/', views.room_delete, name='room_delete'),
    path('employe_management/', views.employe_management, name='employe_management'),
    path('employe/create/', views.employe_create, name='employe_create'),
    path('employe/<int:employee_id>/update/', views.employe_update, name='employe_update'),
    path('employe/<int:employee_id>/delete/', views.employe_delete, name='employe_delete'),
    path('payment_management/', views.payment_management, name='payment_management'),
    path('payment/mark-as-paid/', views.mark_payment_as_paid, name='mark_payment_as_paid'),
]
