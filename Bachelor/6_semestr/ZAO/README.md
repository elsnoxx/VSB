# Základy analýzy obrazu (ZAO)


Seznam ukolu
1. OpenCV + YOLO - Done
2. Analýza semaforů - Done 
3. Vyhledávání vzorů (template matching) - Dart Done
4. Analýza obsazenosti parkoviště pomocí hranových detektorů
5.
6.
7.
8. Počítání dřepů pomocí obrazů - Done


## Lekce 2
Nejdříve převedu obrázek do HSV, jelikož tento formát není tak náchylný na stíny a změny jasu jako RGB. Poté vytvořím masku pro červenou a zelenou barvu – hledaná barva bude v masce bílá a ostatní černé.

Následně provedu dilataci, která roztáhne bílé části a spojí blízké kousky barev k sobě. Poté aplikuji erozi, abych odstranil izolované tečky a zbavil se šumu.

Nakonec spočítám počet červených pixelů a vydělím ho součtem červených a zelených bodů. Tím dostanu poměr (podíl), a pokud je větší než 0.8 (80 %), vyhodnotím barvu jako červenou, v opačném případě jako zelenou

## Lekce 4

1. Načtení a transformace (Perspektiva)
Nejdříve si připravím obraz. Pomocí funkcí order_points a four_point_transform dokážu vzít šikmý pohled na parkoviště a narovnat ho do tzv. "ptačí perspektivy". To usnadňuje práci, protože parkovací místa pak mají pravidelný tvar obdélníku.

2. Předzpracování obrazu
Obrázek převedu na stupně šedi a aplikuji algoritmus CLAHE, který lokálně vylepšuje kontrast (aby byla auta vidět i ve stínu). Poté obraz lehce rozmažu (Gaussian Blur), abych se zbavil drobného šumu.

3. Detekce hran (Canny Edge Detection)
Klíčem k detekci je funkce edges_from_gray. Ta hledá v obrázku hrany (ostré přechody mezi světlou a tmavou).

Auto má na sobě spoustu detailů (okna, světla, SPZ), takže tvoří hodně hran.

Prázdné parkovací místo (asfalt) je hladké a má hran minimum.

4. Kontrola parkovacích míst
Funkce check_spots prochází jednotlivá místa definovaná v mapě:

Vytvoří masku pro jedno konkrétní parkovací místo.

Spočítá, kolik „pixelů hran“ se nachází uvnitř tohoto místa.

Vypočítá hustotu hran (počet hran / plocha místa).

Pokud je hustota větší než nastavený práh (occ_thresh), místo se označí jako obsazené (červeně), jinak jako volné (zeleně).

## Lekce 5

1. Hledání obličeje (Zepředu i z profilu)
Nejdříve program převede obraz do šedi a hledá obličej. Používá k tomu dvě kaskády: jednu pro pohled zepředu a druhou pro profil. Tím zajistí, že řidiče najde, i když trochu otočí hlavu. Používá se tu filtr na váhu (weight_threshold), aby se eliminovaly falešné detekce (např. aby program nepovažoval opěrku hlavy za obličej).

2. Oříznutí oblasti zájmu (ROI)
Když program najde obličej, nevytěžuje procesor prohledáváním celého obrazu. Vytvoří si tzv. ROI (Region of Interest) a oči hledá už jen „uvnitř“ nalezeného obličeje. To obrovsky zrychluje výpočet.

3. Detekce očí a intenzita světla
V oblasti obličeje se spustí další hledání, tentokrát zaměřené na oči. Jakmile oko najde, podívá se na průměrný jas pixelů v této malé oblasti:

Otevřené oko: Obsahuje více světlých ploch (bělmo, odlesky), takže má vyšší průměrný jas (nad prahem 80).

Zavřené oko: Vidíme jen víčko, které je tmavší a matnější, takže má nižší jas.

4. Vyhodnocení a přesnost
Program v reálném čase porovnává svůj odhad (open/close) se skutečným stavem zapsaným v souboru eye-state.txt.

Měří čas detekce (jak dlouho trvalo zpracovat jeden snímek).

Na konci vypíše celkovou přesnost (accuracy) v procentech, takže hned vidíš, jak dobře tvůj model funguje.

## Lekce 6



Website: https://mrl.cs.vsb.cz//people/fusek/zao_course.html


https://github.com/MAKVSB/ZAO/blob/main/lec2/game1.py
http://github.com/jakubcernik/VSB-ZAO/blob/main/Lecture3/main.py