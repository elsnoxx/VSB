#include <iostream>
#include <string>
#include <ctime>

#include "KeyValue.h"
#include "Email.h"
#include "Osoba.h"
#include "Dokument.h"

//Úkoly na cvičení
//• Navrhněte, deklarujte a definujte jednoduché třídy a napište kód, který pracuje s objekty této třídy.
//• E-mail
//• Osoba
//• Dokument

using namespace std;

int main()
{
    KeyValue kv1(1, 1.5);
    cout << kv1.GetValue() << endl;
    cout << endl;
    KeyValue* kv2 = new KeyValue(2, 2.5);
    cout << kv2->GetValue() << endl;
    delete kv2;

    // getchar(); // Toto můžeš odstranit, pokud není potřeba
    cout << endl;
    Email* email = new Email(2, "vsb.cz", "fic0024");
    cout << email->GetId() << " " << email->GetName() << "@" << email->GetDomain() << endl;
    delete email;

    cout << endl;
    Osoba osoba("Richard", "Ficek", "0101155549");
    cout << "Jmeno: " << osoba.GetJmeno() << " Prijmeni: " << osoba.GetPrijmeni() << " Datum narozeni: " << osoba.GetDatumNarozeni() << endl;
    cout << endl;
    Dokument dokument("Faktura", 1001);

    cout << "Typ dokumentu: " << dokument.GetTyp() << " Datum vytvoreni: " << dokument.GetDatumVytvoreni() << " Cislo dokumentu: " << dokument.GetCislo()  << endl;

    


    return 0;
}
