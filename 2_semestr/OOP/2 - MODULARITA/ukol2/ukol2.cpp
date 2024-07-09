#include <iostream>

#include "KeyValue.h"
#include "StringKeyValue.h"

//Úkoly na cvičení
//• Implementujte třídu KeyValue dle přednášky a vytvořte zřetězenou lineární strukturu mnoha(např.tisíce) objektů a pracujte s ní(vypište
//	např.všechny klíče od prvního do posledního objektu).
// 
//• Vytvořte podobnou třídu jako KeyValue, ale s hodnotou i klíčem typu
//	(třídy) string a se dvěma sousedícími(next) objekty.Výsledkem bude tzv. strom.
// 
//• Implementuje strukturu(rozhodovací strom) na identifikaci zvířat nebo
//	rostlin.Klíčem uzlu stromu je rozhodovací kritérium, hodnotou uzlu je
//	název zvířete nebo rostliny, resp.druhu apod.Naplňte klíč alespoň deseti
//	objekty a vypište celou strukturu na obrazovku.

using namespace std;

int main()
{
	KeyValue* kv1 = new KeyValue(1, 1.5);
	cout << kv1->CreateNext(2, 2.5) << endl;

	KeyValue* kv2 = kv1->GetNext();
	cout << kv2->GetNext() << endl;

	//delete kv2;
	delete kv1;

	cout << kv1->GetKey() << endl;
	cout << kv2->GetKey() << endl;


	// Vytvoření prvního objektu
	KeyValue* kv3 = new KeyValue(1, 1.5);
	KeyValue* current = kv3;

	// Vytvoření zřetězené struktury dalších objektů
	for (int i = 2; i <= 1000; ++i) {
		current = current->CreateNext(i, i * 1.5);
	}

	current = kv3;
	while (current != nullptr) {
		cout << "Key: " << current->GetKey() << ", Value: " << current->GetValue() << endl;
		current = current->GetNext();
	}
	delete kv3;


	StringKeyValue* root = new StringKeyValue("root", "root_value");
	root->CreateLeft("left", "left_value");
	root->CreateRight("right", "right_value");

	cout << "Root key: " << root->GetKey() << ", value: " << root->GetValue() << endl;
	cout << "Left child key: " << root->GetLeft()->GetKey() << ", value: " << root->GetLeft()->GetValue() << endl;
	cout << "Right child key: " << root->GetRight()->GetKey() << ", value: " << root->GetRight()->GetValue() << endl;

	delete root;

	// Vytvoření kořene stromu
	StringKeyValue* root2 = new StringKeyValue("Je to savec?", "Rozhodnuti");

	// Přidání dalších uzlů do stromu
	StringKeyValue* node1 = root2->CreateLeft("Ma to pruhy?", "Rozhodnuti");
	StringKeyValue* node2 = root2->CreateRight("Je to rostlina?", "Rozhodnuti");

	node1->CreateLeft("Tygr", "Zvire");
	node1->CreateRight("Kůň", "Zvire");

	StringKeyValue* node3 = node2->CreateLeft("Je to strom?", "Rozhodnuti");
	node2->CreateRight("Sedmikraska", "Rostlina");

	node3->CreateLeft("Dub", "Strom");
	node3->CreateRight("Kapradina", "Rostlina");

	// Další uzly pro naplnění stromu
	root2->GetLeft()->GetLeft()->CreateLeft("Bengalsky tygr", "Zvire");
	root2->GetLeft()->GetLeft()->CreateRight("Sibirský tygr", "Zvire");
	root2->GetLeft()->GetRight()->CreateLeft("Arabsky kun", "Zvire");
	root2->GetLeft()->GetRight()->CreateRight("Plnokrevnik", "Zvire");

	// Další uzly pro dosažení minimálně deseti objektů
	root2->GetRight()->GetLeft()->GetLeft()->CreateLeft("Dub letni", "Strom");
	root2->GetRight()->GetLeft()->GetLeft()->CreateRight("Dub zimni", "Strom");
	root2->GetRight()->GetLeft()->GetRight()->CreateLeft("Kapradina samici", "Rostlina");
	root2->GetRight()->GetLeft()->GetRight()->CreateRight("Kapradina muzska", "Rostlina");

	// Výpis celého stromu
	cout << "Rozhodovaci strom:" << endl;
	root2->PrintTree();

	delete root2;
}
