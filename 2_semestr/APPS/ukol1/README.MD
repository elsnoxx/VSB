# Zadání úkolu č.1

Ve všech příkladech využívejte globální proměnné. Pro první 4 příklady použijte jen instrukce MOV, MOVSX a MOVZX. Od příkladu 5 pak používejte aritmetické, bitové a skokové instrukce.


1. Otočte pořadí znaků v řetězci délky 5 znaků: ``` char zvire[] = "kocka";```.

2. Rozložte proměnnou ```int xnum = 0x1F2E3D4C na``` jednotlivé bajty do proměnných char bajt0, bajt1, bajt2, bajt3;. Vypisujte výsledek v hex formátu, aby byl vidět správný výsledek.

3. Programátor si naivně uložil heslo do pole ```int pass[ 3 ] = { 0x65707573, 0x69727072, 0x33746176 }```. Přesuňte obsah pole pass do pole ```char text[ 16 ] = "my empty string"``` and vytiskněte. Nezapomeňte na ukončovací znak řetězce.

4. V poli ```pass``` z předchozího příkladu vynulujte všem prvkům horní bajt. Vypište výsledek v hex formátu.

5. Vytvořte si pole typu long minimálně délky 10 a inicializujte si ho kladnými i zápornými čísly. Napište funkci, která ze všech čísel v poli udělá čísla sudá: ```pole[ i ] &= ~1```.

6. Otočte všem číslům z předchozího příkladu znaménko. Nepoužívejte instrukci ```NEG```.

7. Vytvořte si pole typu int délky minimálně 10 a inicializujte ho kladnými i zápornými čísly. Napište funkci, která najde minimální prvek v poli a vynuluje ho.

8. Vytvořte si pole typu ```char cisla[ 10 ] = { -10, -20, -30, -40, -50, -60, -70, -80, -90, -100 }```. Proveďte součet prvků pole do proměnné int tak, aby nedošlo k přetečení při sčítání. Ověřte kód funkce i na kladných číslech.