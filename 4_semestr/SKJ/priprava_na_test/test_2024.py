
"""
Úkol 1 (15 bodů)

Jednotka SG-11 se ztratila! Je třeba pro ni vyslat záchrannou misi.
Bohužel není jasné, kde přesně se ztratila, k dispozici jsou pouze útržky nouzových signálů z různých bodů,
kde by jednotka mohla být.

Pomozte najít jednotku SG-11 tím, že naimplementujete funkci spocti_krychli.
Funkce ze souboru na zadané cestě načte 3D souřadnice bodů a najde nejmenší krychli, která ohraničuje tyto
souřadnice (stačí naleznout minimální/maximální souřadnice na všech osách).
Každý řádek v souboru reprezentuje jeden 3D bod, jednotlivé souřadnice jsou odděleny čárkou.
Funkce poté vrátí objem tohoto krychle, aby šlo zjistit, jak velký prostor je nutný prohledat k nalezení jednotky.

Příklad (ukázka souboru soubor_souradnice):
1,2,3
-5,8,-20
2,6,0

spocti_krychli("souradnice_test.txt") # 966
# minimum první souřadnice je -5, maximum je 2, takže první rozměr je 7, obdobně lze naleznout zbylé rozměry
"""
def spocti_krychli(soubor_souradnice):
    
    souradnice = []

    with open(soubor_souradnice) as soubor:
        for radek in soubor:
            temp = radek.strip().split(",")
            souradnice.append((int(temp[0]), int(temp[1]), int(temp[2])))

    souradnice = sorted(souradnice, key=lambda x: x[0])
    x = int(souradnice[len(souradnice)-1][0]) - int(souradnice[0][0])

    souradnice = sorted(souradnice, key=lambda x: x[1])
    y = int(souradnice[len(souradnice)-1][1]) - int(souradnice[0][1])

    souradnice = sorted(souradnice, key=lambda x: x[2])
    z = int(souradnice[len(souradnice)-1][2]) - int(souradnice[0][2])

    return x * y * z

#print(spocti_krychli("souradnice_test.txt"))
"""
Úkol 2 (10 bodů)

Pro nalezení SG-11 je třeba vyřešit další problém - nefunguje ovládací panel k Hvězdné bráně.
Naštěstí Samantha Carterová navrhla dočasné řešení - pomozte jí vytvořit softwarový modul pro ovládání brány.

Naimplementujte třídu OvladaciPanel, která obdrží seznam znaků na panelu.
Znaky jsou uspořádané do kruhu, jeden ze znaků je vždy aktivní (na začátku to bude nultý znak v seznamu).
Uživatel může měnit aktivní znak pomocí pohybu doleva nebo doprava.
Nezapomeňte, že znaky jsou uspořádány do kruhu, lze je tak všechny projet pohybem pouze doleva nebo pouze doprava.

Uživatel může aktivní znak zadat, čímž dojde k přidání aktivního znaku do adresy.
Po přidání všech požadovaných znaků může vytočit adresu (při vytočení panel vrátí seznam všech navolených
znaků v pořadí, ve kterém byly navoleny).

Příklad:
panel = OvladaciPanel(["A", "B", "C"]) # nejprve je aktivní znak "A"
panel.posun_doprava() # nyní je aktivní znak "B"
panel.zadej_znak()    # znak "B" je přidán do vytočené adresy
panel.posun_doprava() # nyní je aktivní znak "C"
panel.posun_doprava() # nyní je aktivní znak "A"
panel.zadej_znak()    # znak "A" je přidán do vytočené adresy
panel.posun_doleva()  # nyní je aktivní znak "C"
panel.zadej_znak()    # znak "C" je přidán do vytočené adresy
panel.vytoc_adresu()  # vrátí ["B", "A", "C"]
"""
class OvladaciPanel:

    def __init__(self, znaky) -> None:
        self.aktivni_znak = 0
        self.znaky_brana = znaky
        self.vytocena_adesa = []

    def posun_doprava(self):
        if len(self.znaky_brana) - 1 < self.aktivni_znak + 1:
            self.aktivni_znak = 0
        else: 
            self.aktivni_znak = self.aktivni_znak + 1

    def posun_doleva(self):
        if self.aktivni_znak - 1 < 0:
            self.aktivni_znak = len(self.znaky_brana) - 1
        else: 
            self.aktivni_znak = self.aktivni_znak - 1

    def zadej_znak(self):
        self.vytocena_adesa.append(self.znaky_brana[self.aktivni_znak])

    def vytoc_adresu(self):
        return self.vytocena_adesa


"""
Úkol 3 (15 bodů)

Nyní už víme přibližnou lokaci SG-11 a máme funkční ovládací panel, zbývá určit přesnou polohu jednotky.
V operačním deníku jednotky jsou adresy planet, které SG-11 navštívila, z těchto informací a přibližné lokace
jednotky by mělo jít dohledat, kde se jednotka přesně nachází.

Naimplementujte funkci nejcastejsi_glyfy.
Funkce ze souboru na zadané cestě načte adresy planet, kam SG-11 cestovala.
Každá adresa (řádek v souboru) je tvořena několika slovy (glyfy) oddělenými mezerou.
Spočítejte, kolikrát se jednotlivé glyfy vyskytují v souboru a vraťte z funkce seznam dvojic (glyf, počet výskytů)
seřazený sestupně dle počtu výskytů jednotlivých glyfů.
Pokud budou mít dva nebo více glyfů stejný počet výskytů, seřaďte je lexikograficky vzestupně dle jejich názvu
("dle abecedy").
V jedné adrese se konkrétní glyf může vyskytovat maximálně jednou.

Příklad (ukázka souboru soubor_adresy):
Crater Taurus Virgo Capricornus Auriga Eridanus Gemini
Taurus Crater Lynx Hydra Auriga Sagittarius Orion
Crater Aries Taurus Scutum Sagittarius Gemini Norma

nejcastejsi_glyfy("planety_test.txt")
# [
#  ('Crater', 3), ('Taurus', 3), ('Auriga', 2), ('Gemini', 2), ('Sagittarius', 2), ('Aries', 1),
#  ('Capricornus', 1), ('Eridanus', 1), ('Hydra', 1), ('Lynx', 1), ('Norma', 1), ('Orion', 1), ('Scutum', 1),
#  ('Virgo', 1)
# ]
"""

def nejcastejsi_glyfy(soubor_adresy):
    
    set_planet = {}
    pole_planet = []

    slovnik_pocty = {}

    with open(soubor_adresy) as soubor:
        for radek in soubor:
            temp = radek.strip().split(" ")
            for planeta in temp:
                pole_planet.append(planeta)

    set_planet = set(pole_planet)


    for item in set_planet:
        for prvky_pole in pole_planet:
            if item == prvky_pole:
                if prvky_pole in slovnik_pocty:
                    slovnik_pocty[prvky_pole] = slovnik_pocty[prvky_pole] + 1
                else:
                    slovnik_pocty[prvky_pole] = 1
            
    pole_test = []
    for klic, hodnota in slovnik_pocty.items():
        pole_test.append(tuple((klic, hodnota)))
    

    sort_temp = sorted(pole_test, key=lambda x: x[0])
    sort_temp = sorted(sort_temp, key=lambda x: x[1], reverse=True)


    return sort_temp

print(nejcastejsi_glyfy("planety_test.txt"))
