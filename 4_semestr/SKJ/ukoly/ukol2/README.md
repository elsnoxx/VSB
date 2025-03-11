## Práce s funkcemi, iterátory, generátory a dekorátory

Ukážeme si, jak pracovat s funkcemi, iterátory, generátory a dekorátory.

Odpovídající materiály naleznete:
- [Základní kurz](https://naucse.python.cz/course/pyladies/)
- [Generátory](https://naucse.python.cz/course/mi-pyt/advanced/generators/)
- [Magické metody](https://naucse.python.cz/course/mi-pyt/intro/magic/)

### Materiály
- [cheatsheet.py](/task/SKJ/2022S/BER0134/ex02_fn/asset/assets/cheatsheet.py) – tahák se soubory, výjimkami, funkcemi, generátory a dekorátory
- [tasks.py](/task/SKJ/2022S/BER0134/ex02_fn/asset/assets/tasks.py) – zadání úloh na cvičení
- [tests.py](/task/SKJ/2022S/BER0134/ex02_fn/asset/assets/tests.py) – testy k úlohám na cvičení

### Domácí úkoly
- [tasks.py](/task/SKJ/2022S/BER0134/ex02_fn/asset/template/tasks.py) – zadání DÚ
- [tests.py](/task/SKJ/2022S/BER0134/ex02_fn/asset/template/tests.py) – testy k DÚ

Řešení domácí úlohy odevzdávejte zde do Kelvina (stačí odevzdat vyřešený soubor `tasks.py`).  
Za úlohy můžete získat maximálně **5 bodů**.

### Nastavení prostředí a testů
Při používání Pythonu si vždy vytvořte virtuální prostředí (zatím bude stačit jedno sdílené pro SKJ).  
Nainstalujte si do něj knihovnu `pytest`, abyste mohli spouštět připravené unit testy.

```bash
$ python3 -m venv venv/skj          # Vytvoří virtuální prostředí (spusťte pouze jednou)
$ source venv/skj/bin/activate      # Aktivuje virtuální prostředí (spusťte po zapnutí terminálu)
(skj) $ pip install pytest          # Nainstaluje pytest (spusťte pouze jednou)
(skj) $ python -m pytest tests.py   # Spustí testy ze souboru tests.py
```
