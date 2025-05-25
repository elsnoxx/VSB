from datetime import date
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth import authenticate, login, logout
from django.http import JsonResponse
from django.contrib import messages
from .models import Guest, Reservation, Room, Employee, Payment, ServiceUsage
from .forms import GuestForm, ReservationForm, RoomForm, EmployeeForm, CustomUserCreationForm, GuestRegisterForm
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm, UserChangeForm
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render
from .models import Reservation, Feedback, Guest
from .forms import FeedbackForm
from django.views.decorators.http import require_POST
from django.contrib.admin.views.decorators import staff_member_required
from .models import Service, ServiceUsage


#################################################
# AUTENTIZACE A UŽIVATELSKÉ ÚČTY
#################################################
def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('index')
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('login')

def register_view(request):
    if request.method == 'POST':
        form = GuestRegisterForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.email = form.cleaned_data['email']
            user.first_name = form.cleaned_data['firstname']
            user.last_name = form.cleaned_data['lastname']
            user.save()
            Guest.objects.create(
                user=user,
                firstname=form.cleaned_data['firstname'],
                lastname=form.cleaned_data['lastname'],
                email=form.cleaned_data['email'],
                phone=form.cleaned_data['phone'],
                birth_date=form.cleaned_data['birth_date'],
                street=form.cleaned_data['street'],
                city=form.cleaned_data['city'],
                postal_code=form.cleaned_data['postal_code'],
                country=form.cleaned_data['country'],
                guest_type='normal',
                notes=form.cleaned_data['notes']
            )
            messages.success(request, 'Registrace proběhla úspěšně. Nyní se můžete přihlásit.')
            return redirect('login')
    else:
        form = GuestRegisterForm()
    return render(request, 'register.html', {'form': form})


#################################################
# UŽIVATELSKÝ PROFIL
#################################################

@login_required
def profile_view(request):
    try:
        guest = Guest.objects.get(user=request.user)
    except Guest.DoesNotExist:
        guest = Guest.objects.create(
            user=request.user,
            firstname=request.user.first_name,
            lastname=request.user.last_name,
            email=request.user.email,
            birth_date="2000-01-01",
            street="",
            city="",
            postal_code="",
            country="",
            guest_type="normal"
        )
    
    return render(request, 'profile.html', {'user': request.user, 'guest': guest})

@login_required
def profile_edit(request):
    if request.method == 'POST':
        form = UserChangeForm(request.POST, instance=request.user)
        if form.is_valid():
            form.save()
            messages.success(request, 'Profil byl upraven.')
            return redirect('profile')
    else:
        form = UserChangeForm(instance=request.user)
    return render(request, 'profile_edit.html', {'form': form})

@login_required
def my_reservations(request):
    reservations = Reservation.objects.filter(guest__user=request.user)
    return render(request, 'management/reservation/reservation_list.html', {'reservations': reservations, 'my_only': True})

@login_required
def profile_update(request):
    if request.method == 'POST':
        guest = get_object_or_404(Guest, user=request.user)
        form_type = request.POST.get('form_type')
        
        try:
            if form_type == 'personal':
                # Debug výpis
                print(f"Před aktualizací: {guest.firstname}, {guest.lastname}, {guest.email}, {guest.phone}, {guest.birth_date}")
                
                guest.firstname = request.POST.get('firstname')
                guest.lastname = request.POST.get('lastname')
                guest.email = request.POST.get('email')
                guest.phone = request.POST.get('phone')
                guest.birth_date = request.POST.get('birth_date')
                
                # Aktualizace související údaje uživatelského účtu
                request.user.first_name = request.POST.get('firstname')
                request.user.last_name = request.POST.get('lastname')
                request.user.email = request.POST.get('email')
                request.user.save()
                
                # Debug výpis po aktualizaci
                print(f"Po aktualizaci: {guest.firstname}, {guest.lastname}, {guest.email}, {guest.phone}, {guest.birth_date}")
                
            elif form_type == 'address':
                guest.street = request.POST.get('street')
                guest.city = request.POST.get('city')
                guest.postal_code = request.POST.get('postal_code')
                guest.country = request.POST.get('country')
                
            elif form_type == 'notes':
                guest.notes = request.POST.get('notes')
            
            guest.save()
            messages.success(request, 'Profil byl úspěšně aktualizován.')
        except Exception as e:
            messages.error(request, f'Chyba při aktualizaci profilu: {str(e)}')
            # Debug výpis chyby
            import traceback
            traceback.print_exc()
        
        return redirect('profile')
    return redirect('profile')

#################################################
# HLAVNÍ STRÁNKA
#################################################
def index(request):
    rooms = Room.objects.all()
    
    floors = {
        "1": [],
        "2": [],
        "3": []
    }

    for room in rooms:
        floor_number = str(room.room_number)[0]
        if floor_number in floors:
            floors[floor_number].append(room)
    
    return render(request, 'management/reservation/reservation_rooms.html', {'floors': floors})


#################################################
# SPRÁVA HOSTŮ
#################################################

@login_required
def guest_list(request):
    guests = Guest.objects.all()
    return render(request, 'management/guest/guest_list.html', {'guests': guests})

# Detail hosta
def guest_detail(request, guest_id):
    guest = get_object_or_404(Guest, pk=guest_id)
    return render(request, 'management/guest/guest_detail.html', {'guest': guest})

@login_required
# Vytvoření hosta
def guest_create(request):
    if request.method == 'POST':
        guest_form = GuestForm(request.POST)
        if guest_form.is_valid():
            guest = guest_form.save(commit=False)
            guest.guest_type = 'normal'  # nastav výchozí typ hosta
            guest.save()
            return redirect('guest_list')
        else:
            return render(request, 'management/guest/guest_form.html', {'form': guest_form})
    else:
        guest_form = GuestForm()
    return render(request, 'management/guest/guest_form.html', {'form': guest_form})

@login_required
def add_guest(request):
    if request.method == 'POST':
        guest_form = GuestForm(request.POST)
        if guest_form.is_valid():
            guest = guest_form.save(commit=False)
            guest.guest_type = 'normal'
            guest.save()
            return redirect('guest_list')
    else:
        guest_form = GuestForm()
    return render(request, 'management/guest/guest_form.html', {'form': guest_form})

@login_required
# Editace hosta
def guest_update(request, guest_id):
    guest = get_object_or_404(Guest, pk=guest_id)
    if request.method == "POST":
        form = GuestForm(request.POST, instance=guest)
        if form.is_valid():
            form.save()
            return redirect('guest_detail', guest_id=guest.pk)
    else:
        form = GuestForm(instance=guest)
    return render(request, 'management/guest/guest_form.html', {'form': form})


@login_required
# Smazání hosta
def guest_delete(request, guest_id):
    guest = get_object_or_404(Guest, pk=guest_id)
    if request.method == "POST":
        guest.delete()
        return JsonResponse({'success': True})  # Vrátí JSON odpověď
    return JsonResponse({'success': False}, status=400)


#################################################
# SPRÁVA REZERVACÍ
#################################################
@login_required
def create_reservation(request, room_id):
    room = get_object_or_404(Room, pk=room_id)
    guest, created = Guest.objects.get_or_create(
        user=request.user,
        defaults={
            "firstname": request.user.first_name or "Neznámý",
            "lastname": request.user.last_name or "Uživatel",
            "email": request.user.email or "",
            "birth_date": "2000-01-01",
            "street": "",
            "city": "",
            "postal_code": "",
            "country": "",
            "guest_type": "normal"
        }
    )
    from datetime import date, timedelta
    default_check_in = date.today()
    default_check_out = date.today() + timedelta(days=1)
    price = room.room_type.price_per_night

    if request.method == 'POST':
        form = ReservationForm(request.POST)
        if form.is_valid():
            reservation = form.save(commit=False)
            reservation.guest = guest
            reservation.room = room
            reservation.accommodation_price = price
            reservation.status = "Nová"
            # Vytvoř nový payment a přiřaď ho k rezervaci
            from .models import Payment
            payment = Payment.objects.create(
                total_accommodation=price,
                total_expenses=0
            )
            reservation.payment = payment
            reservation.save()
            # Nastav pokoj jako obsazený
            room.is_occupied = True
            room.save()
            return redirect('reservation_detail', reservation_id=reservation.pk)
    else:
        form = ReservationForm(initial={
            'check_in_date': default_check_in,
            'check_out_date': default_check_out,
        })
    return render(request, 'create_reservation.html', {
        'form': form,
        'room': room,
        'guest': guest,
        'price': price,
        'default_check_in': default_check_in,
        'default_check_out': default_check_out,
    })

@login_required
def reservation_list(request):
    reservations = Reservation.objects.all()
    if request.user.is_superuser:
        guest = request.GET.get('guest')
        check_in = request.GET.get('check_in')
        if guest:
            reservations = reservations.filter(guest__firstname__icontains=guest) | reservations.filter(guest__lastname__icontains=guest)
        if check_in:
            reservations = reservations.filter(check_in_date=check_in)
    else:
        reservations = reservations.filter(guest__user=request.user)
    return render(request, 'management/reservation/reservation_list.html', {'reservations': reservations})


@login_required
def reservation_detail(request, reservation_id):
    reservation = get_object_or_404(Reservation, pk=reservation_id)
    guest = reservation.guest
    room = reservation.room
    payment = reservation.payment
    is_admin = request.user.is_staff or request.user.is_superuser

    
    all_services = Service.objects.all()
    used_services = ServiceUsage.objects.filter(reservation=reservation).select_related('service')

    total_services_price = sum([us.total_price for us in used_services])
    
    feedback_exists = Feedback.objects.filter(reservation=reservation, guest=guest).exists()
    feedback_list = Feedback.objects.filter(reservation=reservation)
    
    context = {
        'reservation': reservation,
        'guest': guest,
        'room': room,
        'payment': payment,
        'is_admin': is_admin,
        'all_services': all_services,
        'used_services': used_services,
        'total_services_price': total_services_price,
        'feedback_exists': feedback_exists,
        'feedback_list': feedback_list,
    }
    return render(request, 'management/reservation/reservation_detail.html', context)

@login_required
def reservation_update(request, reservation_id):
    reservation = get_object_or_404(Reservation, pk=reservation_id)
    if request.method == "POST":
        form = ReservationForm(request.POST, instance=reservation)
        if form.is_valid():
            form.save()
            return redirect('reservation_list')
    else:
        form = ReservationForm(instance=reservation)
    return render(request, 'management/reservation/reservation_form.html', {'form': form})


@login_required
def reservation_delete(request, reservation_id):
    reservation = get_object_or_404(Reservation, pk=reservation_id)
    payment = reservation.payment
    

    if request.method == "POST":
        reservation.delete()
        payment.delete() if payment else None
        return redirect('reservation_list')
    return render(request, 'management/reservation/reservation_confirm_delete.html', {'reservation': reservation})

@login_required
def reservation_success(request, roomid):
    room = get_object_or_404(Room, pk=roomid)
    if room:
        room.is_occupied = True
        room.save()
        return render(request, 'reservation_success.html', {'room': room})

#################################################
# SLUŽBY A POPLATKY
#################################################
@login_required
@require_POST
def add_service_to_reservation(request, reservation_id):
    from .models import Service, ServiceUsage
    reservation = get_object_or_404(Reservation, pk=reservation_id)
    service_id = request.POST.get('service_id')
    quantity = int(request.POST.get('quantity', 1))
    service = get_object_or_404(Service, pk=service_id)
    price = service.price
    total_price = price * quantity

    ServiceUsage.objects.create(
        reservation=reservation,
        service=service,
        quantity=quantity,
        total_price=total_price
    )
    payment = reservation.payment
    payment.total_expenses += total_price
    payment.save()
    return redirect('reservation_detail', reservation_id=reservation_id)

@login_required
@require_POST
def remove_service_from_reservation(request, reservation_id, usage_id):
    usage = get_object_or_404(ServiceUsage, pk=usage_id, reservation_id=reservation_id)
    
    payment = usage.reservation.payment
    payment.total_expenses -= usage.total_price
    payment.save()
    usage.delete()
    return redirect('reservation_detail', reservation_id=reservation_id)

@staff_member_required
def service_management(request):
    from .models import Service
    from .forms import ServiceForm
    services = Service.objects.all()
    if request.method == "POST":
        form = ServiceForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('service_management')
    else:
        form = ServiceForm()
    return render(request, 'management/service/service_management.html', {'services': services, 'form': form})


#################################################
# PLATBY
#################################################
@login_required
def mark_payment_as_paid_from_reservation(request, reservation_id):
    if request.method == "POST":
        reservation = get_object_or_404(Reservation, pk=reservation_id)
        payment = reservation.payment
        if payment:
            payment.is_paid = True
            payment.payment_date = date.today()
            payment.save()
            room = reservation.room
            room.is_occupied = False
            room.save()
            reservation.status = 'Closed'
            reservation.save()
        return redirect('reservation_detail', reservation_id=reservation_id)
    return JsonResponse({'error': 'Invalid request method'}, status=400)

@login_required
def payment_management(request):
    payment = Payment.objects.all()
    return render(request, 'management/payment/payment_management.html', {'payment': payment})

@login_required
@csrf_exempt
def mark_payment_as_paid(request, employee_id):
    if request.method == "POST":
        payment = get_object_or_404(Payment, pk=employee_id)
        payment.is_paid = True
        payment.payment_date = date.today()
        payment.save()
        
        reservations = Reservation.objects.filter(payment=payment)
        for reservation in reservations:
            room = reservation.room
            room.is_occupied = False
            room.save()
        return redirect('payment_management')
    return JsonResponse({'error': 'Invalid request method'}, status=400)

#################################################
# SPRÁVA POKOJŮ
#################################################

@login_required
def room_occupancies(request):
    return render(request, 'room_occupancies.html')

@login_required
def room_management(request):
    rooms = Room.objects.select_related('room_type').all()
    return render(request, 'management/room/room_management.html', {'rooms': rooms})


@login_required
def room_create(request):
    if request.method == 'POST':
        form = RoomForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('room_management')
    else:
        form = RoomForm()
    return render(request, 'forms/room_form.html', {'form': form})

@login_required
def room_update(request, room_id):
    room = get_object_or_404(Room, pk=room_id)
    if request.method == "POST":
        form = RoomForm(request.POST, instance=room)
        if form.is_valid():
            form.save()
            return redirect('room_management')
    else:
        form = RoomForm(instance=room)
    return render(request, 'forms/room_form.html', {'form': form})


@csrf_exempt
@login_required
def room_delete(request, room_id):
    if request.method == "POST":
        room = get_object_or_404(Room, pk=room_id)
        room.delete()
        return redirect('room_management') 


#################################################
# SPRÁVA POKOJŮ
#################################################

@login_required
def employe_management(request):
    emp = Employee.objects.all()
    return render(request, 'management/employe/employe_management.html', {'emp': emp})

@login_required
def employe_create(request):
    if request.method == 'POST':
        form = EmployeeForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('employe_management')
    else:
        form = EmployeeForm()
    return render(request, 'forms/employe_form.html', {'form': form})

@login_required
def employe_update(request, employee_id):
    employee = get_object_or_404(Employee, pk=employee_id)
    if request.method == "POST":
        form = EmployeeForm(request.POST, instance=employee)
        if form.is_valid():
            form.save()
            return redirect('employe_management')
    else:
        form = EmployeeForm(instance=employee)
    return render(request, 'forms/employee_form.html', {'form': form})

@login_required
def employe_delete(request, employee_id):
    employee = get_object_or_404(Employee, pk=employee_id)
    if request.method == "POST":
        employee.delete()
        return redirect('employe_management')
    return render(request, 'management/employe/employe_confirm_delete.html', {'employee': employee})


#################################################
# AUTOCOMPLETE HOSTŮ
#################################################

def guest_autocomplete(request):
    q = request.GET.get('q', '')
    guests = Guest.objects.filter(
        firstname__icontains=q
    ) | Guest.objects.filter(
        lastname__icontains=q
    )
    results = [f"{g.firstname} {g.lastname}" for g in guests.distinct()[:10]]
    return JsonResponse(results, safe=False)

#################################################
# ZPĚTNÁ VAZBA
#################################################
@login_required
@require_POST
def api_add_feedback(request, reservation_id):
    reservation = get_object_or_404(Reservation, pk=reservation_id)
    guest = get_object_or_404(Guest, user=request.user)
    if reservation.status not in ['Closed', 'Ukončena'] or reservation.guest != guest:
        return JsonResponse({'error': 'Not allowed'}, status=403)
    if Feedback.objects.filter(reservation=reservation, guest=guest).exists():
        return JsonResponse({'error': 'Feedback already exists'}, status=400)
    form = FeedbackForm(request.POST)
    if form.is_valid():
        feedback = form.save(commit=False)
        feedback.guest = guest
        feedback.reservation = reservation
        feedback.save()
        return JsonResponse({
            'success': True,
            'rating': feedback.rating,
            'comment': feedback.comment,
            'date': feedback.feedback_date.strftime('%d.%m.%Y')
        })
    return JsonResponse({'error': 'Invalid data'}, status=400)

