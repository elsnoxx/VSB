# Vlákna APPS

Výsledky řešní naleznete v [vysledky.md](2_semestr\APPS\thread-ukol\vysledky.md).

## **Zadání**

Implementujte třídění pole čísel s pomocí vláken.

Využijte jeden ze tří algoritmů dle následujícího klíče: `login_uživatele % 3`, tedy např. `oli10 -> 10 % 3 = 1` a algoritmy jsou 3 v poli [ přímý_výběr, přímé vkládání, bubble_sort ]. Pro `oli10` tedy přímé vkládání.

Dále sudý login bude mít čísla `long` a třídění vzestupně a lichý login `int` a třídění sestupně.

1. Napište si program, ve kterém si ověříte, že třídící algoritmus funguje správně a **hlavně** třídí správně jen zadanou část pole!
2. Upravte program tak, aby se při spuštění jako parametry musela zadat délka pole a počet vláken pro třídění.
3. V programu se nejprve vygenerují náhodná čísla v rozsahu < -10<sup>9</sup> - 10<sup>9</sup> > a vytvoří se kopie pole.
4. Provede se setřídění prvního pole v jednom vlákně a změří se čas třídění.
5. Druhé pole se rozdělí na požadovaný počet dílů a každý díl se bude následně třídit ve vlastním vlákně (paralelně s ostatními vlákny). Změří se čas třídění.
6. Jednotlivé setříděné části se spojí v jeden výsledek (např. [1](https://www.youtube.com/results?search_query=merging+sorted+array), [2](https://www.youtube.com/watch?v=xF3TU-QlhJQ)). Spojování ideálně vždy jen po dvojicích, nemá smysl spojovat 3, 4 a více částí současně. Spojování má složitost O(N) a čas spojování je zanedbatelný ve srovnání s časem třídění. Čas spojování se změří.
7. Porovnání prvního a druhého výsledku třídění, zda jsou výsledky shodné.
8. Finální měření času proveďte na počítači na učebně pro 2-3-4-5-6 vláken. Stačí se k počítači připojit vzdáleně. Délku pole nastavte tak, aby třídění bez vláken trvalo alespoň 30s. Naměřené hodnoty z jednotlivých
