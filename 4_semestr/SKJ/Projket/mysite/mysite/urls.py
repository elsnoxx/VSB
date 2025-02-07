from django.urls import path
from django.contrib import admin
from blog.views import (
    index, guest_list, guest_detail, guest_create, guest_update, guest_delete,
    reservation_list, reservation_detail,add_guest
)

urlpatterns = [
    path('', index, name='index'),
    path('admin/', admin.site.urls),
    path('guests/', guest_list, name='guest_list'),
    path('guests/<int:guest_id>/', guest_detail, name='guest_detail'),
    path('guests/new/', guest_create, name='guest_create'),
    path('guests/<int:guest_id>/edit/', guest_update, name='guest_update'),
    path('guests/<int:guest_id>/delete/', guest_delete, name='guest_delete'),
    path('reservations/', reservation_list, name='reservation_list'),
    path('reservations/<int:reservation_id>/', reservation_detail, name='reservation_detail'),
    path('add-guest/', add_guest, name='add_guest'),
]
