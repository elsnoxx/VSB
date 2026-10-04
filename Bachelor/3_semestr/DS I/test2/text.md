# Přehled DDL Klíčových Slov a Datových Typů v SQL

## Hlavní klíčová slova v DDL

- **CREATE** - Používá se k vytvoření nových databázových objektů (tabulek, databází, pohledů, indexů atd.).
  - *Příklad*: `CREATE TABLE`, `CREATE DATABASE`, `CREATE INDEX`

- **ALTER** - Slouží ke změnám existujících objektů, jako je přidávání nebo mazání sloupců.
  - *Příklad*: `ALTER TABLE`, `ALTER DATABASE`

- **DROP** - Smaže databázový objekt, jako je tabulka, index nebo pohled.
  - *Příklad*: `DROP TABLE`, `DROP INDEX`

- **TRUNCATE** - Rychle vymaže všechna data z tabulky, ale zachová její strukturu.
  - *Příklad*: `TRUNCATE TABLE`

- **RENAME** - Používá se pro přejmenování existujících objektů (převážně tabulek).
  - *Příklad*: `RENAME TABLE old_name TO new_name`

- **COMMENT** - Umožňuje přidávat komentáře k databázovým objektům pro lepší dokumentaci.
  - *Příklad*: `COMMENT ON TABLE table_name IS 'description'`

- **REFERENCES** - Používá se při definování cizích klíčů k odkazování na jiné tabulky.
  - *Příklad*: `FOREIGN KEY (column) REFERENCES other_table (column)`

---

## Základní datové typy v SQL

SQL nabízí různé datové typy, které lze využít v rámci DDL při vytváření tabulek.

### Číselné typy:
- `INT`, `INTEGER` - Celé číslo.
- `SMALLINT` - Menší celé číslo.
- `BIGINT` - Velké celé číslo.
- `DECIMAL(p, s)` - Desetinné číslo s přesností (p - počet číslic, s - desetinných míst).
- `NUMERIC(p, s)` - Podobný jako `DECIMAL`.
- `FLOAT`, `REAL` - Čísla s plovoucí desetinnou čárkou.

### Textové typy:
- `CHAR(n)` - Pevná délka řetězce o délce `n`.
- `VARCHAR(n)` - Řetězec proměnné délky, maximálně `n` znaků.
- `TEXT` - Velký textový řetězec bez limitu délky.

### Datum a čas:
- `DATE` - Datum ve formátu `YYYY-MM-DD`.
- `TIME` - Čas ve formátu `HH:MM:SS`.
- `DATETIME` - Datum a čas dohromady.
- `TIMESTAMP` - Datum a čas s automatickým nastavením na aktuální čas při vložení nebo aktualizaci.
- `INTERVAL` - Specifikuje časové období.

### Logické typy:
- `BOOLEAN` - Logická hodnota `TRUE` nebo `FALSE`.

### Binární typy:
- `BINARY`, `VARBINARY` - Binární data, např. pro ukládání obrázků nebo souborů.

### Ostatní datové typy (mohou se lišit podle databáze):
- `ENUM` - Výčet možností (např. `ENUM('male', 'female')`).
- `SET` - Podobné jako `ENUM`, ale umožňuje více hodnot najednou.
- `JSON` - Datový typ pro ukládání JSON struktury.
- `UUID` - Univerzálně unikátní identifikátor.
- `XML` - Pro ukládání XML dat.

---

## Další klíčová slova a omezení (constraints)

- **PRIMARY KEY** - Definuje primární klíč tabulky, který jednoznačně identifikuje každý řádek.
- **FOREIGN KEY** - Vytváří cizí klíč pro vztahy mezi tabulkami.
- **UNIQUE** - Zajišťuje, že sloupec nebo kombinace sloupců bude mít unikátní hodnoty.
- **NOT NULL** - Nastaví sloupec jako povinný (nesmí obsahovat `NULL` hodnotu).
- **CHECK** - Umožňuje nastavit podmínku, která musí být splněna.
- **DEFAULT** - Nastaví výchozí hodnotu pro sloupec.
- **AUTO_INCREMENT** (MySQL) nebo **SERIAL** (PostgreSQL) - Automaticky zvyšuje hodnotu u primárního klíče.
