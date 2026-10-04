# Řešení projektu

## 1. Zadání (jiné než z přednášek) si vymyslíte sami, projekt ale musí obsahovat minimálně sedm tříd.
V tomto projektu je navržen a implementován systém pro autoservis. Systém se skládá z různých tříd, které modelují pracovníky, zákazníky, vozidla, součástky, zakázky a faktury. Projekt obsahuje minimálně sedm tříd, využívá kompozice objektů, přetížené metody, dědičnost a polymorfismus.

## 2.  Vlastní návrh popsaný v textové podobě a doplněný UML diagramem tříd

UML diagram se nachází v repozitáři.

### Popis návrhu
Návrh projektu obsahuje následující hlavní třídy:

1. **AbstractOsoba**: Abstraktní třída reprezentující obecnou osobu.
2. **AbstractPracovnik**: Abstraktní třída reprezentující obecného pracovníka, dědí z AbstractOsoba.
3. **Mechanik**, Asistent, Ucetni: Konkrétní třídy dědící z AbstractPracovnik, představují různé typy pracovníků.
4. **Zakaznik**: Třída reprezentující zákazníka, dědí z AbstractOsoba.
5. **Vozidlo**: Třída reprezentující vozidlo, obsahuje objekt třídy Zakaznik.
6. **Soucastka**: Třída reprezentující součástku vozidla.
7. **Zakazka**: Třída reprezentující zakázku na opravu vozidla, obsahuje objekt třídy Vozidlo.
8. **Faktura**: Třída reprezentující fakturu za opravu, obsahuje objekt třídy Zakazka.
9. **Autoservis**: Třída reprezentující autoservis, obsahuje kolekce pracovníků, zakázek a faktur.

## 3. Projekt bude obsahovat kompozice (hierarchie) objektů

Kompozice je použita v třídě Vozidlo, která obsahuje objekt třídy Zakaznik.

```cpp
class Vozidlo {
private:
    string znacka;
    string model;
    string vin;
    Zakaznik* zakaznik; // Kompozice

public:
    Vozidlo(string znacka, string model, string vin, Zakaznik* zakaznik)
        : znacka(znacka), model(model), vin(vin), zakaznik(zakaznik) {}
};
```

## 4. Projekt bude obsahovat a používat přetížené metody

Přetížené metody jsou použity v konstruktoru třídy AbstractPracovnik.

```cpp
class AbstractPracovnik : public AbstractOsoba {
public:
    AbstractPracovnik(string jmeno, string prijmeni) : AbstractOsoba(jmeno, prijmeni) {}
    AbstractPracovnik(string jmeno, string prijmeni, bool dovolena) : AbstractOsoba(jmeno, prijmeni) {
        this->dovolena = dovolena;
    }
};
```



## 5. Projekt bude obsahovat a používat třídu v roli objektu

Třída Zakaznik je používána jako objekt v třídě Vozidlo.

```cpp
class Zakaznik : public AbstractOsoba {
private:
    string adresa;
    string telefon;

public:
    Zakaznik(string jmeno, string prijmeni, string adresa, string telefon)
        : AbstractOsoba(jmeno, prijmeni), adresa(adresa), telefon(telefon) {}
};

Vozidlo* vozidlo = new Vozidlo("Skoda", "Octavia", "VIN123456", zakaznik); // Zakaznik jako objekt v Vozidlo
```

## 6. Projekt bude obsahovat a využívat dědičnou hierarchii obsahující alespoň tři třídy

Dědičná hierarchie mezi třídami AbstractOsoba, AbstractPracovnik, Mechanik, Asistent a Ucetni.

```cpp
class AbstractOsoba {
protected:
    string jmeno;
    string prijmeni;

public:
    AbstractOsoba(string jmeno, string prijmeni) : jmeno(jmeno), prijmeni(prijmeni) {}
    virtual ~AbstractOsoba() = default;
};

class AbstractPracovnik : public AbstractOsoba {
protected:
    bool dovolena;

public:
    AbstractPracovnik(string jmeno, string prijmeni) : AbstractOsoba(jmeno, prijmeni), dovolena(false) {}
    AbstractPracovnik(string jmeno, string prijmeni, bool dovolena) : AbstractOsoba(jmeno, prijmeni), dovolena(dovolena) {}
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
    void pracovat() override { std::cout << "Ucetni vytvari fakturu. \n"; }
};
```

## 7. Dědičnost bude obsahovat a používat jak rozšíření (dat i metod), tak změnu chování s využitou pozdní vazbou

Třída Mechanik rozšiřuje třídu AbstractPracovnik a přepisuje metodu pracovat.

```cpp
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
    void pracovat() override { std::cout << "Ucetni vytvari fakturu. \n"; }
};
```

## 8. Dědičná hierarchie bude obsahovat čistě abstraktní třídu

AbstractPracovnik je abstraktní třída s čistě abstraktní metodou pracovat.

```cpp
class AbstractPracovnik : public AbstractOsoba {
public:
    AbstractPracovnik(string jmeno, string prijmeni) : AbstractOsoba(jmeno, prijmeni), dovolena(false) {}
    AbstractPracovnik(string jmeno, string prijmeni, bool dovolena) : AbstractOsoba(jmeno, prijmeni), dovolena(dovolena) {}
    virtual void pracovat() = 0; // Čistě abstraktní metoda
};
```

## 9. Projekt bude využívat polymorfismus (polymorfní přiřazení i polymorfní datovou strukturu)

Polymorfní datová struktura je použita ve třídě Autoservis, která obsahuje vektor ukazatelů na AbstractPracovnik.

```cpp
class Autoservis {
private:
    vector<AbstractPracovnik*> pracovnici; // Polymorfní datová struktura

public:
    void pridatPracovnika(AbstractPracovnik* pracovnik) {
        pracovnici.push_back(pracovnik); // Polymorfní přiřazení
    }
};
```
## 10. Po spuštění projektu bude vytvořeno alespoň několik desítek objektů zahrnujících objekty všech deklarovaných tříd (s výjimkou abstraktních) a ve výpisu na konzole budou prezentovány výsledky úloh, které objekty vykonají

V hlavní funkci main je vytvořeno několik objektů a jsou prezentovány výsledky jejich úloh.

```cpp
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
    Vozidlo* vozidlo1 = new Vozidlo("Skoda", "Octavia", "123456789", new Zakaznik("Petr", "Svoboda", "Praha", "123456789"));
    Vozidlo* vozidlo2 = new Vozidlo("Volkswagen", "Golf", "987654321", new Zakaznik("Jana", "Novakova", "Brno", "987654321"));

    // vytváříme objekty součástek
    Soucastka* predniBrzdy = new Soucastka("Predni brzdy", 5000);
    Soucastka* predniSvetlo = new Soucastka("Predni svetlo", 1000);
    Soucastka* filtr = new Soucastka("Kabinový filtr", 500);
    Soucastka* zadniBrzdy = new Soucastka("Zadni brzdy", 3000);

    // vytváříme objekty zakázek
    Zakazka* zakazka1 = new Zakazka(vozidlo1, mechanik1);
    Zakazka* zakazka2 = new Zakazka(vozidlo2, mechanik2);
    Zakazka* zakazka3 = new Zakazka(vozidlo1, mechanik2);
    zakazka1->pridatSoucastku(predniBrzdy);
    zakazka1->pridatSoucastku(predniSvetlo);
    zakazka1->pridatSoucastku(filtr);
    zakazka2->pridatSoucastku(zadniBrzdy);
    zakazka2->pridatSoucastku(filtr);
    zakazka3->pridatSoucastku(zadniBrzdy);

    // vytváříme objekty faktur
    Faktura* faktura1 = new Faktura(zakazka1, 6500);
    Faktura* faktura2 = new Faktura(zakazka2, 3500);

    // přidáváme zakázky a faktury do autoservisu
    autoservis.pridatZakazku(zakazka1);
    autoservis.pridatZakazku(zakazka2);
    autoservis.pridatZakazku(zakazka3);
    autoservis.pridatFakturu(faktura1);
    autoservis.pridatFakturu(faktura2);

    // vypisujeme výsledky
    autoservis.prace();
    autoservis.vypisZakazky();
    autoservis.vypisFaktury();
    autoservis.zaplaceneFaktury();
    autoservis.dokonceneZaplaceneFaktury();

    // čištění paměti
    delete mechanik1;
    delete mechanik2;
    delete asistent;
    delete ucetni;
    delete vozidlo1;
    delete vozidlo2;
    delete predniBrzdy;
    delete predniSvetlo;
    delete filtr;
    delete zadniBrzdy;
    delete zakazka1;
    delete zakazka2;
    delete zakazka3;
    delete faktura1;
    delete faktura2;

    return 0;
}
```
