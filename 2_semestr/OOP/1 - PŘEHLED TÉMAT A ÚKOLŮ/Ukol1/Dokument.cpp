#include "Dokument.h"
#include <ctime>

Dokument::Dokument(string typ, int cislo) {
    this->typ = typ;
    this->cislo = cislo;
    time(&this->datum_vytvoreni); // Nastavení aktuálního ?asu p?i vytvo?ení objektu
}

string Dokument::GetTyp() {
    return this->typ;
}

string Dokument::GetDatumVytvoreni() {
    char buffer[80];
    struct tm timeinfo;
#ifdef _WIN32
    localtime_s(&timeinfo, &this->datum_vytvoreni);
#else
    localtime_r(&this->datum_vytvoreni, &timeinfo);
#endif
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &timeinfo);
    return string(buffer);
}

int Dokument::GetCislo() {
    return this->cislo;
}
