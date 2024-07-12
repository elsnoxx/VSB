# Test z ASM

Řešen pouze hlavní temín a ne oprava.


## **Hlavní termín**

Volání funkcí s parametry.

1. Spočítejte, kolik čísel v poli je v zadaném rozsahu.

```c
   int in_range(long *tp_array, int t_n, long t_from, long t_to);
```

2. Vyplňte pole mocninami čísla X. Při přetečení budou další výsledky 0.

```c
   void mocniny(long *tp_pole, int t_N, int t_X);
```

3. Ověřte, zda je zadané číslo X mocninou čísla M. Výsledek bude -1 nebo příslušná mocnina.

```c
   int je_mocnina(int t_X, int t_M);
```

4. Které číslo v poli char má nejmenší počet jedniček?

```c
   int nejednicky(char *tp_pole, int t_N);
```


## **Oprava**

Volání funkcí s parametry.

1. Spočítejte, kolik znaků v řetězci je v zadaném rozsahu. Například pro text `programovani pocitace` znaky od `'a'` do `'g'`.

```c
   int in_range(char *tp_str, char t_od, char t_do);
```

2. Vyplňte pole aritmetickou posloupností od pocatek se zadanou hodnotou krok.

```c
   void aritmeticka(int *t_pole, int t_len, int t_pocatek, int t_krok);
```

3. Kolikrát lze zadané číslo dělit číslem K beze zbytku?

```c
   int kolikrat(int t_cislo, int t_K);
```

4. Který bajt v čísle obsahuje největší hodnotu? Např. pro číslo 0x1133557788664422 to bude bajt číslo 3 (nejnižší je 0).

```c
   int maxbyte(long t_cislo);
```


