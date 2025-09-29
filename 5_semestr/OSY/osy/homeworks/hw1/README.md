# Poznámky k úkolu

## Úkoly

### Úkol 1: Základní implementace
- **Cíl:** Implementovat dva programy:
  - `gennum`: Generuje čísla účtů podle pravidel ČNB.
  - `verbank`: Ověřuje platnost čísel účtů.
- **Použité funkce:**
  - `printf`/`scanf` pro textový vstup a výstup.
  - `read`/`write` pro binární vstup a výstup.
- **Testování:**
  - `./gennum 3000030000 > numbers.txt`
  - `./verbank < numbers.txt`

### Úkol 2A: Statická knihovna
- **Cíl:** Přesunout funkci pro generování čísel do samostatného souboru a vytvořit z ní statickou knihovnu.
- **Implementace:**
  - `generator.cpp` obsahuje funkci pro generování čísel.
  - Program `gennum` linkuje statickou knihovnu.

### Úkol 2B: Dynamická knihovna
- **Cíl:** Přesunout funkci pro validaci čísel účtů do samostatného souboru a vytvořit z ní dynamickou knihovnu.
- **Implementace:**
  - `validateNumber.cpp` obsahuje funkci pro validaci čísel účtů.
  - Program `verbank` linkuje dynamickou knihovnu.

### Úkol 2C: Přepínání dynamických knihoven
- **Cíl:** Vytvořit dvě dynamické knihovny se stejným názvem, ale s různými implementacemi:
  - `libvalidateNumber.so` pro čísla účtů.
  - `libvalidateNumber.so` pro rodná čísla.
- **Implementace:**
  - Pomocí `LD_LIBRARY_PATH` lze přepínat mezi knihovnami.
  - Testování:
    - Pro čísla účtů:
      ```bash
      export LD_LIBRARY_PATH=$(pwd)/lib
      ./bin/verbank < numbers.txt
      ```
    - Pro rodná čísla:
      ```bash
      export LD_LIBRARY_PATH=$(pwd)/rodnecislo
      ./bin/verbank < numbers.txt
      ```
---
