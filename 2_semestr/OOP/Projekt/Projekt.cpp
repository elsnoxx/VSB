//1.  Zadání(jiné než z přednášek) si vymyslíte sami, projekt ale musí obsahovat minimálně sedm tříd.
//2.  Součástí projektu bude vlastní návrh popsaný v textové podobě a doplněný UML diagramem tříd(nebo jiným schématem popisujícím srozumitelně vztahy mezi třídami).
//3.  Projekt bude obsahovat kompozice(hierarchie) objektů.
//4.  Projekt bude obsahovat a používat přetížené metody.
//5.  Projekt bude obsahovat a používat třídu v roli objektu.
//6.  Projekt bude obsahovat a využívat dědičnou hierarchii obsahující alespoň tři třídy.
//7.  Dědičnost bude obsahovat a používat jak rozšíření(dat i metod), tak změnu chování s využitou pozdní vazbou.
//8.  Dědičná hierarchie bude obsahovat čistě abstraktní třídu.
//9.  Projekt bude využívat polymorfismus(polymorfní přiřazení i polymorfní datovou strukturu).
//10. Po spuštění projektu bude vytvořeno alespoň několik desítek objektů zahrnujících objekty všech deklarovaných tříd(s výjimkou abstraktních) a ve výpisu na konzole budou prezentovány výsledky úloh, které objekty vykonají.

#include <iostream>
#include <string>
#include <vector>

using namespace std;

class AbstractOsoba {
private:
    string jmeno;
    string prijmeni;

public:
    AbstractOsoba(string jmeno, string prijmeni) : jmeno(jmeno), prijmeni(prijmeni) {}

    string getFullName() { return jmeno + " " + prijmeni; }
};

class AbstractPracovnik : public AbstractOsoba {
private:
    bool dovolena = false;

public:
    AbstractPracovnik(string jmeno, string prijmeni) : AbstractOsoba(jmeno, prijmeni) {}
    AbstractPracovnik(string jmeno, string prijmeni, bool dovolena) : AbstractOsoba(jmeno, prijmeni) {
        this->dovolena = dovolena;
    }

    bool getDovolena() { return dovolena; }
    virtual void pracovat() = 0;
};

class Mechanik : public AbstractPracovnik {
public:
    Mechanik(string jmeno, string prijmeni) : AbstractPracovnik(jmeno, prijmeni) {}
    Mechanik(string jmeno, string prijmeni, bool dovolena) : AbstractPracovnik(jmeno, prijmeni, dovolena) {}

    void pracovat() override {
        std::cout << "Mechanik pracuje na oprave vozidla.\n";
    }
};

class Asistent : public AbstractPracovnik {
public:
    Asistent(string jmeno, string prijmeni) : AbstractPracovnik(jmeno, prijmeni) {}
    Asistent(string jmeno, string prijmeni, bool dovolena) : AbstractPracovnik(jmeno, prijmeni, dovolena) {}

    void pracovat() override { std::cout << "Asistent kontroluje vozidlo.\n"; }
};

class Ucetni : public AbstractPracovnik {
public:
    Ucetni(string jmeno, string prijmeni) : AbstractPracovnik(jmeno, prijmeni) {}
    Ucetni(string jmeno, string prijmeni, bool dovolena) : AbstractPracovnik(jmeno, prijmeni, dovolena) {}

    void pracovat() override { std::cout << "Ucetni vytvari fakturu.\n"; }
};

class Zakaznik : public AbstractOsoba {
private:
    string adresa;
    string telefon;

public:
    Zakaznik(string jmeno, string prijmeni, string adresa, string telefon)
        : AbstractOsoba(jmeno, prijmeni), adresa(adresa), telefon(telefon) {}

    string getDetails() {
        return "Adresa: " + adresa + ", telefon: " + telefon;
    }
};

class Vozidlo {
private:
    string znacka;
    string model;
    string vin;
    Zakaznik* zakaznik;

public:
    Vozidlo(string znacka, string model, string vin, Zakaznik* zakaznik)
        : znacka(znacka), model(model), vin(vin), zakaznik(zakaznik) {}

    string getDetails() {
        return znacka + ", " + model + ", VIN: " + vin;
    }

    Zakaznik* getZakaznik() { return zakaznik; }
};

class Soucastka {
private:
    string nazev;
    int cena;

public:
    Soucastka(string nazev, int cena) : nazev(nazev), cena(cena) {}

    int getCena() { return cena; }

    string getDetails() {
        return "Nazev: " + nazev + ", cena: " + to_string(cena);
    }
};

class Zakazka {
private:
    Vozidlo* vozidlo;
    Mechanik* mechanik;
    vector<Soucastka*> soucastky;
    bool dokoncena = false;
    int cena = 0;

public:
    Zakazka(Vozidlo* vozidlo, Mechanik* mechanik) : vozidlo(vozidlo), mechanik(mechanik) {}

    void pridatSoucastku(Soucastka* soucastka) {
        soucastky.push_back(soucastka);
        cena += soucastka->getCena();
    }

    int getCena() { return cena; }

    bool getDokoncena() { return dokoncena; }
    void setDokoncena(bool dokoncena) { this->dokoncena = dokoncena; }


    string getDetails() {
        string details =
            "Zakaznik: " + vozidlo->getZakaznik()->getFullName() + "\n";
        details += "Mechanik: " + mechanik->getFullName() + "\n";
        details += "Vozidlo: " + vozidlo->getDetails() + "\n";
        details += "Soucastky:\n";
        for (auto soucastka : soucastky) {
            details += " - " + soucastka->getDetails() + "\n";
        }
        details += "Cena: " + to_string(cena) + "\n";

        details += "Dokoncena: ";
        if (dokoncena)
            details += "Ano\n";
        else
            details += "Ne\n";
        
        return details;
    }
};


class Faktura {
private:
    Zakazka* zakazka;
    int cena;
    bool zaplacena = false;

public:
    Faktura(Zakazka* zakazka) {
        this->zakazka = zakazka;
        cena = zakazka->getCena();
    }

    void zaplatit() {
        zaplacena = true;
    }

    bool jeZaplacena() const {
        return zaplacena;
    }

    Zakazka* getZakazka() const {
        return zakazka;
    }

    string getDetails() {
        string details = zakazka->getDetails() + "\n";
        details += "Cena: " + to_string(cena) + "\n";
        details += "Zaplacena: ";
        if (jeZaplacena())
            details += "ano\n";
        else
            details += "ne\n";

        return details;
    }
};


class Autoservis {
private:
    vector<AbstractPracovnik*> pracovnici;
    vector<Zakazka*> zakazky;
    vector<Faktura*> faktury;

public:
    void pridatPracovnika(AbstractPracovnik* pracovnik) { pracovnici.push_back(pracovnik); }

    void pridatZakazku(Zakazka* zakazka) { zakazky.push_back(zakazka); }

    void pridatFakturu(Faktura* faktura) { faktury.push_back(faktura); }

    void prace() {
        for (auto pracovnik : pracovnici) {
            if (!pracovnik->getDovolena())
                pracovnik->pracovat();
        }
    }

    void vypisZakazky() {
        cout << "Nedokoncene zakazky:" << endl;
        for (auto zakazka : zakazky) {
            if (!zakazka->getDokoncena())
                cout << zakazka->getDetails() << endl;
        }
    }

    void vypisDokonceneZakazky() {
        cout << "Dokoncene zakazky:" << endl;
        for (auto zakazka : zakazky) {
            if (zakazka->getDokoncena())
                cout << zakazka->getDetails() << endl;
        }
    }

    void vypisFaktury() {
        cout << "Faktury:" << endl;
        for (auto faktura : faktury) {
            cout << faktura->getDetails() << endl;
        }
    }

    void zaplaceneFaktury() {
        cout << "Faktury:" << endl;
        for (auto faktura : faktury) {
            if (faktura->jeZaplacena()) {
                cout << faktura->getDetails() << endl;
            }
        }
    }

    void dokonceneZaplaceneFaktury() const {
        cout << "Dokončené a zaplacené faktury:" << endl;
        for (auto faktura : faktury) {
            Zakazka* zakazka = faktura->getZakazka();
            if (zakazka->getDokoncena() && faktura->jeZaplacena()) {
                cout << faktura->getDetails() << endl;
            }
        }
    }
};


int main() {
    Autoservis autoservis;

    // vytváříme objekty pracovníků
    Mechanik* mechanik1 = new Mechanik("Jan", "Novak");
    Mechanik* mechanik2 = new Mechanik("Marek", "Fiala", true);
    Asistent* asistent = new Asistent("Richard", "Ficek");
    Ucetni* ucetni = new Ucetni("Jana", "Novakova");
    autoservis.pridatPracovnika(mechanik1);
    autoservis.pridatPracovnika(mechanik2);
    autoservis.pridatPracovnika(asistent);
    autoservis.pridatPracovnika(ucetni);

    // vytváříme objekty vozidel
    Vozidlo* vozidlo1 =
        new Vozidlo("Skoda", "Octavia", "123456789",
            new Zakaznik("Petr", "Svoboda", "Praha", "123456789"));
    Vozidlo* vozidlo2 =
        new Vozidlo("Volkswagen", "Golf", "987654321",
            new Zakaznik("Jana", "Novakova", "Brno", "987654321"));
    Vozidlo* vozidlo3 =
        new Vozidlo("Renault", "Clio", "789456123",
            new Zakaznik("Michal", "Novotny", "Ostrava", "654987321"));
    Vozidlo* vozidlo4 =
        new Vozidlo("Toyota", "Corolla", "654123987",
            new Zakaznik("Tereza", "Kovarova", "Plzen", "789321654"));
    Vozidlo* vozidlo5 =
        new Vozidlo("Ford", "Focus", "159753468",
            new Zakaznik("Marie", "Dvorakova", "Liberec", "456789321"));
    Vozidlo* vozidlo6 =
        new Vozidlo("Chevrolet", "Camaro", "357951468",
            new Zakaznik("Jan", "Kolar", "Kolin", "987654123"));
    Vozidlo* vozidlo7 =
        new Vozidlo("BMW", "X5", "456123789",
            new Zakaznik("Eva", "Novakova", "Olomouc", "789654123"));
    Vozidlo* vozidlo8 =
        new Vozidlo("Mercedes", "E-Class", "369852147",
            new Zakaznik("Martin", "Svoboda", "Ceske Budejovice", "654123789"));


    // vytváříme objekty součástek
    Soucastka* predniBrzdy = new Soucastka("Predni brzdy", 5000);
    Soucastka* predniSvetlo = new Soucastka("Predni svetlo", 1000);
    Soucastka* filtr = new Soucastka("Kabinový filtr", 500);
    Soucastka* zadniBrzdy = new Soucastka("Zadni brzdy", 3000);
    Soucastka* baterie = new Soucastka("Baterie", 2000);
    Soucastka* vyfukovePotrubie = new Soucastka("Výfukové potrubí", 1500);
    Soucastka* palivovyFiltr = new Soucastka("Palivový filtr", 800);
    Soucastka* chladic = new Soucastka("Chladič", 2500);
    Soucastka* olejovyFiltr = new Soucastka("Olejový filtr", 600);

    // vytváříme objekty zakázek
    Zakazka* zakazka1 = new Zakazka(vozidlo1, mechanik1);
    Zakazka* zakazka2 = new Zakazka(vozidlo2, mechanik2);
    Zakazka* zakazka3 = new Zakazka(vozidlo3, mechanik2);
    Zakazka* zakazka4 = new Zakazka(vozidlo4, mechanik1);
    Zakazka* zakazka5 = new Zakazka(vozidlo5, mechanik2);
    Zakazka* zakazka6 = new Zakazka(vozidlo6, mechanik1);
    Zakazka* zakazka7 = new Zakazka(vozidlo7, mechanik2);
    Zakazka* zakazka8 = new Zakazka(vozidlo8, mechanik1);


    zakazka1->pridatSoucastku(predniBrzdy);
    zakazka1->pridatSoucastku(predniSvetlo);
    zakazka1->pridatSoucastku(filtr);

    zakazka2->pridatSoucastku(zadniBrzdy);
    zakazka2->pridatSoucastku(filtr);

    zakazka3->pridatSoucastku(zadniBrzdy);

    zakazka4->pridatSoucastku(vyfukovePotrubie);

    zakazka5->pridatSoucastku(palivovyFiltr);

    zakazka6->pridatSoucastku(chladic);
    zakazka6->pridatSoucastku(olejovyFiltr);

    zakazka7->pridatSoucastku(predniBrzdy);
    zakazka7->pridatSoucastku(filtr);

    zakazka8->pridatSoucastku(predniSvetlo);
    zakazka8->pridatSoucastku(baterie);

    // vytváříme objekty faktur
    Faktura* faktura1 = new Faktura(zakazka1);
    Faktura* faktura2 = new Faktura(zakazka2);
    Faktura* faktura3 = new Faktura(zakazka3);
    Faktura* faktura4 = new Faktura(zakazka4);
    Faktura* faktura5 = new Faktura(zakazka5);
    Faktura* faktura6 = new Faktura(zakazka6);
    Faktura* faktura7 = new Faktura(zakazka7);
    Faktura* faktura8 = new Faktura(zakazka8);

    autoservis.pridatFakturu(faktura1);
    autoservis.pridatFakturu(faktura2);
    autoservis.pridatFakturu(faktura3);
    autoservis.pridatFakturu(faktura4);
    autoservis.pridatFakturu(faktura5);
    autoservis.pridatFakturu(faktura6);
    autoservis.pridatFakturu(faktura7);
    autoservis.pridatFakturu(faktura8);

    // Nastavení informací o zaplacení a platbě
    faktura1->zaplatit();
    faktura3->zaplatit();
    faktura5->zaplatit();
    faktura7->zaplatit();

    // dokoncene zakazky
    zakazka3->setDokoncena(true);
    zakazka7->setDokoncena(true);
    

    // přidáváme zakázky a faktury do autoservisu
    autoservis.pridatZakazku(zakazka1);
    autoservis.pridatZakazku(zakazka2);
    autoservis.pridatZakazku(zakazka3);
    autoservis.pridatZakazku(zakazka4);
    autoservis.pridatZakazku(zakazka5);
    autoservis.pridatZakazku(zakazka6);
    autoservis.pridatZakazku(zakazka7);
    autoservis.pridatZakazku(zakazka8);

    


    // vypisujeme výsledky
    autoservis.prace();
    cout << endl;
    cout << endl;
    cout << endl;
    cout << endl;
    cout << endl;
    autoservis.vypisZakazky();
    cout << endl;
    cout << endl;
    cout << endl;
    cout << endl;
    cout << endl;
    autoservis.vypisFaktury();
    cout << endl;
    cout << endl;
    cout << endl;
    cout << endl;
    cout << endl;
    autoservis.vypisDokonceneZakazky();
    cout << endl;
    cout << endl;
    cout << endl;
    cout << endl;
    cout << endl;
    autoservis.zaplaceneFaktury();
    cout << endl;
    cout << endl;
    cout << endl;
    cout << endl;
    cout << endl;
    autoservis.dokonceneZaplaceneFaktury();

    // Dealokace paměti
    delete mechanik1;
    delete mechanik2;
    delete asistent;
    delete ucetni;
    delete vozidlo1;
    delete vozidlo2;
    delete vozidlo3;
    delete vozidlo4;
    delete vozidlo5;
    delete vozidlo6;
    delete vozidlo7;
    delete vozidlo8;
    delete predniBrzdy;
    delete predniSvetlo;
    delete filtr;
    delete zadniBrzdy;
    delete baterie;
    delete vyfukovePotrubie;
    delete palivovyFiltr;
    delete chladic;
    delete olejovyFiltr;
    delete zakazka1;
    delete zakazka2;
    delete zakazka3;
    delete zakazka4;
    delete zakazka5;
    delete zakazka6;
    delete zakazka7;
    delete zakazka8;
    delete faktura1;
    delete faktura2;
    delete faktura3;
    delete faktura4;
    delete faktura5;
    delete faktura6;
    delete faktura7;
    delete faktura8;


    return 0;
}
