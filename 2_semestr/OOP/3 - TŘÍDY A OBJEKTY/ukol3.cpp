#include <iostream>
#include <string>
#include <vector>


#include "KeyValue.h"
#include "KeyValues.h"
#include "Faktura.h"
#include "PolozkaFaktury.h"
#include "Osoba.h"

using namespace std;


int main()
{
	int N = 5;
	KeyValues* myKeyValues = new KeyValues(2);

	KeyValue* myKeyValue = myKeyValues->CreateObject(0, 0.5);
	cout << myKeyValue->GetValue() << endl;

	for (int  i = 0; i < N; i++)
	{
		myKeyValues->CreateObject(i, i + 0.5);
	}
	cout << myKeyValues->SearchObject(4)->GetValue() << endl;
	
	KeyValue* removedObject = myKeyValues->RemoveObject(3);

	
	cout << endl;

    Osoba zakaznik("Richard Ficek", "Ostrava, Czech Republic");
    Faktura faktura(123, zakaznik);

    faktura.pridatPolozku(PolozkaFaktury("Produkt A", 100.0));
    faktura.pridatPolozku(PolozkaFaktury("Produkt B", 55.5));
    faktura.pridatPolozku(PolozkaFaktury("Produkt B", 50.0));

    cout << "Celkova cena faktury: " << faktura.celkovaCena() << endl;


    delete myKeyValues;

	return 0;
}