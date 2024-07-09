#pragma once
#ifndef PATIENT_H
#define PATIENT_H

#include <iostream>
#include <string>
#include <vector>




/**
 * @class Patient
 * @brief Třída představuje pacienta s jménem a věkem.
 */
class Patient {
private:
    std::string name;  /**< Jméno pacienta */
    int age;           /**< Věk pacienta */

public:
    /**
     * @brief Konstruktor pro inicializaci pacienta s daným jménem a věkem.
     * @param n Jméno pacienta.
     * @param a Věk pacienta.
     */
    Patient(const std::string& n, int a);

    /**
     * @brief Vrací jméno pacienta.
     * @return Jméno pacienta.
     */
    std::string GetName() const;

    /**
     * @brief Vrací věk pacienta.
     * @return Věk pacienta.
     */
    int GetAge() const;
};

/**
 * @class Doctor
 * @brief Třída představuje lékaře s jménem a specializací.
 */
class Doctor {
private:
    std::string name;            /**< Jméno lékaře */
    std::string specialization;  /**< Specializace lékaře */

public:
    /**
     * @brief Konstruktor pro inicializaci lékaře s daným jménem a specializací.
     * @param n Jméno lékaře.
     * @param s Specializace lékaře.
     */
    Doctor(const std::string& n, const std::string& s);

    /**
     * @brief Vrací jméno lékaře.
     * @return Jméno lékaře.
     */
    std::string GetName() const;

    /**
     * @brief Vrací specializaci lékaře.
     * @return Specializace lékaře.
     */
    std::string GetSpecialization() const;
};

/**
 * @class Appointment
 * @brief Třída představuje schůzku mezi pacientem a lékařem v určitém datu.
 */
class Appointment {
private:
    Patient* patient;       /**< Ukazatel na pacienta */
    Doctor* doctor;         /**< Ukazatel na lékaře */
    std::string date;       /**< Datum schůzky */

public:
    /**
     * @brief Konstruktor pro inicializaci schůzky s daným pacientem, lékařem a datem.
     * @param p Ukazatel na pacienta.
     * @param d Ukazatel na lékaře.
     * @param dt Datum schůzky.
     */
    Appointment(Patient* p, Doctor* d, const std::string& dt);

    /**
     * @brief Vrací ukazatel na pacienta schůzky.
     * @return Ukazatel na pacienta.
     */
    Patient* GetPatient() const;

    /**
     * @brief Vrací ukazatel na lékaře schůzky.
     * @return Ukazatel na lékaře.
     */
    Doctor* GetDoctor() const;

    /**
     * @brief Vrací datum schůzky.
     * @return Datum schůzky.
     */
    std::string GetDate() const;
};

/**
 * @class Clinic
 * @brief Třída představuje kliniku s pacienty, lékaři a schůzkami.
 */
class Clinic {
private:
    std::vector<Patient*> patients;       /**< Vektor pacientů */
    std::vector<Doctor*> doctors;         /**< Vektor lékařů */
    std::vector<Appointment*> appointments;  /**< Vektor schůzek */

public:
    /**
     * @brief Destruktor pro uvolnění paměti alokované pro pacienty, lékaře a schůzky.
     */
    ~Clinic();

    /**
     * @brief Přidá nového pacienta do kliniky.
     * @param name Jméno pacienta.
     * @param age Věk pacienta.
     * @return Ukazatel na nově přidaného pacienta.
     */
    Patient* AddPatient(const std::string& name, int age);

    /**
     * @brief Přidá nového lékaře do kliniky.
     * @param name Jméno lékaře.
     * @param specialization Specializace lékaře.
     * @return Ukazatel na nově přidaného lékaře.
     */
    Doctor* AddDoctor(const std::string& name, const std::string& specialization);

    /**
     * @brief Naplánuje novou schůzku mezi pacientem a lékařem.
     * @param patient Ukazatel na pacienta.
     * @param doctor Ukazatel na lékaře.
     * @param date Datum schůzky.
     * @return Ukazatel na nově naplánovanou schůzku.
     */
    Appointment* ScheduleAppointment(Patient* patient, Doctor* doctor, const std::string& date);

    /**
     * @brief Vypíše všechny naplánované schůzky.
     */
    void PrintAppointments();
};

#endif // PATIENT_H
