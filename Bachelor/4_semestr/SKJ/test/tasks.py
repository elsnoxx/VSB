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
def get_min_max(souradnice):
    min_x = min([bod[0] for bod in souradnice])
    max_x = max([bod[0] for bod in souradnice])
    min_y = min([bod[1] for bod in souradnice])
    max_y = max([bod[1] for bod in souradnice])
    min_z = min([bod[2] for bod in souradnice])
    max_z = max([bod[2] for bod in souradnice])

    return (min_x, max_x), (min_y, max_y), (min_z, max_z)


def spocti_krychli(soubor_souradnice):
    try:
        # zkontrolovat jestli cesta existuje
        if not soubor_souradnice:
            raise ValueError("Cesta k souboru není zadána.")
        
        cooradination = []	

        # otevreni a zpracovani vstupniho souboru
        with open(soubor_souradnice) as file:
            for line in file:
                line = line.strip()
                if line:
                    x, y, z = line.split(",")
                    x = int(x)
                    y = int(y)
                    z = int(z)
                    cooradination.append((x, y, z))

        # pokud jsem nic nenasel nema smysl pokracoavat
        if not cooradination:
            return 0

        # ziskani minimalnich a maximalnich souradnic pro nasledny vypocet
        x, y, z = get_min_max(cooradination)

        # vypocet velikosti v kazdem smeru
        x_size = x[1] - x[0]
        y_size = y[1] - y[0]
        z_size = z[1] - z[0]
        
        
        # vypocet objemu kvadru (hrana na 3)
        volume = x_size * y_size * z_size

        #  vypocet objemu krychle (nejvetsi hrana na 3), pro korektni vystup viz nice pro odkomentovani
        # volume = max(x_size, y_size, z_size) ** 3


        # problem ze test osetruje objem kvadru a ne objem krychle
        return volume

    except FileNotFoundError:
        print(f"Soubor {soubor_souradnice} nebyl nalezen.")
        return None
    except ValueError:
        print(f"Chyba při převodu souřadnic v souboru {soubor_souradnice}.")
        return None
    except Exception as e:
        print(f"Nastala neočekávaná chyba: {e}")
        return None


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
    def __init__(self, symbols):
        self.active_symbol = 0
        self.symbols_list = symbols
        self.dialed_address = []

    # metoda pro ziskani velikosti seznamu
    def length(self):
        return len(self.symbols_list)
    
    # metoda pro ziskanu aktualniho symbolu
    def get_active_symbol(self):
        try:
            return self.symbols_list[self.active_symbol]
        except:
            print("Chyba při získávání aktivního symbolu")
            return None

    # posunuti doprava
    def posun_doprava(self):
        length = self.length()
        if length - 1 < self.active_symbol + 1:
            self.active_symbol = 0
        else:
            self.active_symbol = self.active_symbol + 1

    # posunuti doleva
    def posun_doleva(self):
        length = self.length()
        if self.active_symbol - 1 < 0:
            self.active_symbol = length - 1
        else:
            self.active_symbol = self.active_symbol - 1

    # zadani znaku
    def zadej_znak(self):
        try:
            symbol = self.symbols_list[self.active_symbol]
            self.dialed_address.append(symbol)
        except:
            print("Chyaba při zadávání znaku")

    # vytoceni aktualne nastavene adresy
    def vytoc_adresu(self):
        try:
            return self.dialed_address
        except:
            print("Chyba při vytáčení adresy")
            return []

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
def order_list(seznam):
    for i in range(len(seznam)):
        for j in range(0, len(seznam) - i - 1):
            if (seznam[j][1] < seznam[j + 1][1]) or (seznam[j][1] == seznam[j + 1][1] and seznam[j][0] > seznam[j + 1][0]):
                seznam[j], seznam[j + 1] = seznam[j + 1], seznam[j]
    return seznam

def nejcastejsi_glyfy(soubor_adresy):
    try:
        if not soubor_adresy:
            raise ValueError("Cesta k souboru není zadána.")
        
        glyph_list = []
        glyph_counts = {}

        with open(soubor_adresy) as file:
            for line in file:
                temp = line.strip().split(" ")
                for glyph in temp:
                    glyph_list.append(glyph)

        glyph_set = set(glyph_list)

        for item in glyph_set:
            for glyph_in_list in glyph_list:
                if item == glyph_in_list:
                    if glyph_in_list in glyph_counts:
                        glyph_counts[glyph_in_list] = glyph_counts[glyph_in_list] + 1
                    else:
                        glyph_counts[glyph_in_list] = 1

        result_list = []
        for key, value in glyph_counts.items():
            result_list.append((key, value))


        result_list = order_list(result_list)

        return result_list
    
    except FileNotFoundError:
        print(f"Soubor {soubor_adresy} nebyl nalezen.")
        return None
    except ValueError:
        print(f"Chyba při převodu souřadnic v souboru {soubor_adresy}.")
        return None
    except Exception as e:
        print(f"Nastala neočekávaná chyba: {e}")
        return None




# zavoalni ukolu c. 1
# print(spocti_krychli("souradnice_test.txt"))


# zavoalni ukolu c. 2
# panel = OvladaciPanel(["A", "B", "C"]) # nejprve je aktivní znak "A"
# panel.posun_doprava() # nyní je aktivní znak "B"
# panel.zadej_znak()    # znak "B" je přidán do vytočené adresy
# panel.posun_doprava() # nyní je aktivní znak "C"
# panel.posun_doprava() # nyní je aktivní znak "A"
# panel.zadej_znak()    # znak "A" je přidán do vytočené adresy
# panel.posun_doleva()  # nyní je aktivní znak "C"
# panel.zadej_znak()    # znak "C" je přidán do vytočené adresy
# panel.vytoc_adresu()  # vrátí ["B", "A", "C"]



# zavoalni ukolu c. 3
# print(nejcastejsi_glyfy("planety_test.txt"))
