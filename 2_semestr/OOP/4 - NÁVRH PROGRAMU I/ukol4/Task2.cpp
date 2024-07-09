#include "Task2.h"

// Patient
Patient::Patient(const std::string& n, int a) : name(n), age(a) {}

std::string Patient::GetName() const {
    return name;
}

int Patient::GetAge() const {
    return age;
}

// Docktor

Doctor::Doctor(const std::string& n, const std::string& s) : name(n), specialization(s) {}

std::string Doctor::GetName() const {
    return name;
}

std::string Doctor::GetSpecialization() const {
    return specialization;
}

// Appointment

Appointment::Appointment(Patient* p, Doctor* d, const std::string& dt) : patient(p), doctor(d), date(dt) {}

Patient* Appointment::GetPatient() const {
    return patient;
}

Doctor* Appointment::GetDoctor() const {
    return doctor;
}

std::string Appointment::GetDate() const {
    return date;
}

// Clinic
Clinic::~Clinic() {
    // Uvolnění paměti při destrukci objektu Clinic
    for (Patient* patient : patients) {
        delete patient;
    }
    for (Doctor* doctor : doctors) {
        delete doctor;
    }
    for (Appointment* appointment : appointments) {
        delete appointment;
    }
}

Patient* Clinic::AddPatient(const std::string& name, int age) {
    Patient* patient = new Patient(name, age);
    patients.push_back(patient);
    return patient;
}

Doctor* Clinic::AddDoctor(const std::string& name, const std::string& specialization) {
    Doctor* doctor = new Doctor(name, specialization);
    doctors.push_back(doctor);
    return doctor;
}

Appointment* Clinic::ScheduleAppointment(Patient* patient, Doctor* doctor, const std::string& date) {
    Appointment* appointment = new Appointment(patient, doctor, date);
    appointments.push_back(appointment);
    return appointment;
}

void Clinic::PrintAppointments() {
    std::cout << "Scheduled Appointments:" << std::endl;
    for (Appointment* appointment : appointments) {
        std::cout << "Date: " << appointment->GetDate() << std::endl;
        std::cout << "Doctor: " << appointment->GetDoctor()->GetName() << " ("
            << appointment->GetDoctor()->GetSpecialization() << ")" << std::endl;
        std::cout << "Patient: " << appointment->GetPatient()->GetName()
            << " (Age: " << appointment->GetPatient()->GetAge() << ")" << std::endl;
        std::cout << std::endl;
    }
}