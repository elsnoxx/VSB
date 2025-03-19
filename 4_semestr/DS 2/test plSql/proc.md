# Co je to procedura?
Procedura je uložený blok kódu v databázi, který provádí určitou operaci nebo sadu operací na datech. Může být volána explicitně (např. z aplikace nebo SQL dotazu) nebo automaticky (např. z triggeru).

Procedury se vytvářejí pomocí PL/SQL (Procedural Language/SQL), což je jazyk specifický pro Oracle, který umožňuje kombinovat SQL dotazy s procedurálním kódem.

## Vytvoření procedury
Procedura se vytvoří pomocí příkazu CREATE PROCEDURE. Syntaxe je následující:

```sql
CREATE [OR REPLACE] PROCEDURE nazev_procedury
IS
BEGIN
    -- Tělo procedury
    -- SQL příkazy a PL/SQL kód
END;
```

```sql
CREATE OR REPLACE PROCEDURE zvys_platy IS
BEGIN
    UPDATE zamestnanci
    SET plat = plat * 1.10;
END;
```

## Procedury s parametry
Procedury mohou mít parametry, které umožňují předávat hodnoty do procedury při jejím volání.

IN – Parametr pro příjem hodnoty.
OUT – Parametr pro vrácení hodnoty.
IN OUT – Parametr pro příjem a vrácení hodnoty

```sql
CREATE OR REPLACE PROCEDURE nazev_procedury(param1 IN datatype, param2 OUT datatype) IS
BEGIN
    -- Použití parametrů v těle procedury
END;
```

Příklad:

```sql
CREATE OR REPLACE PROCEDURE zvys_plat_zamestnanci(
    p_id IN NUMBER,
    p_zvyseni IN NUMBER
) IS
BEGIN
    UPDATE zamestnanci
    SET plat = plat + p_zvyseni
    WHERE id = p_id;
END;

EXECUTE zvys_plat_zamestnanci(1, 5000);
```

##  Zpracování výjimek
PL/SQL umožňuje zpracovávat chyby nebo výjimky pomocí bloku EXCEPTION.

Příklad:
```sql
CREATE OR REPLACE PROCEDURE zvys_plat_zamestnanci(p_id IN NUMBER) IS
BEGIN
    UPDATE zamestnanci
    SET plat = plat + 1000
    WHERE id = p_id;
    
    IF SQL%ROWCOUNT = 0 THEN
        RAISE_APPLICATION_ERROR(-20001, 'Zaměstnanec s tímto ID neexistuje.');
    END IF;
EXCEPTION
    WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('Došlo k chybě: ' || SQLERRM);
END;
```
## Drobná vylepšení s dynamickým SQL
Dynamické SQL je užitečné při práci s tabulkami nebo sloupci, jejichž názvy jsou neznámé nebo se mění za běhu aplikace. To umožňuje flexibilní a univerzální řešení.

```sql
CREATE OR REPLACE PROCEDURE ziskej_data(
    p_table_name IN VARCHAR2
) IS
    v_sql VARCHAR2(4000);
BEGIN
    v_sql := 'SELECT * FROM ' || p_table_name;
    EXECUTE IMMEDIATE v_sql;
END;
```