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
    path('room/management/', views.room_management, name='create_reservation')
]
