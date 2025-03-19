# Základní konstrukce v Oracle DB (DDL)

## 1. Vytvoření databázového schématu
Databázové schéma v Oracle se vytváří pomocí `CREATE USER`. Tato konstrukce také umožňuje nastavit počáteční parametry uživatele.

```sql
CREATE USER uzivatel IDENTIFIED BY heslo
DEFAULT TABLESPACE USERS
QUOTA UNLIMITED ON USERS;
```


## 2. Vytvoření tabulky
Tabulky se vytvářejí pomocí příkazu CREATE TABLE. Specifikujeme název tabulky, názvy sloupců a jejich datové typy.

```sql
CREATE TABLE zamestnanci (
    id NUMBER PRIMARY KEY,
    jmeno VARCHAR2(100),
    prijmeni VARCHAR2(100),
    datum_narozeni DATE
);
```


## 3. Přidání sloupce do existující tabulky
Pokud potřebujete přidat sloupec do již existující tabulky, použijte příkaz ALTER TABLE.

```sql
ALTER TABLE zamestnanci
ADD email VARCHAR2(150);
```

## 4. Úprava datového typu sloupce
Pokud chcete změnit datový typ sloupce, použijete příkaz ALTER TABLE.

```sql
ALTER TABLE zamestnanci
MODIFY datum_narozeni TIMESTAMP;
```


## 5. Vytvoření primárního klíče
Primární klíč se obvykle vytváří při definování tabulky, ale může být přidán i později pomocí ALTER TABLE.

```sql
ALTER TABLE zamestnanci
ADD CONSTRAINT pk_zamestnanci PRIMARY KEY (id);
```

## 6. Vytvoření cizího klíče
Cizí klíč propojuje dvě tabulky a zajišťuje referenční integritu. Definuje vztah mezi sloupcem v jedné tabulce a primárním klíčem v jiné tabulce.

```sql
CREATE TABLE projekty (
    id NUMBER PRIMARY KEY,
    nazev VARCHAR2(255),
    zamestnanec_id NUMBER,
    FOREIGN KEY (zamestnanec_id) REFERENCES zamestnanci(id)
);
```

## 7. Odstranění tabulky
Tabulku můžete odstranit pomocí příkazu DROP TABLE. Tento příkaz vymaže tabulku i její data.

```sql
DROP TABLE zamestnanci;
```

## 8. Změna názvu tabulky
Pro změnu názvu tabulky použijeme příkaz RENAME.

```sql
RENAME zamestnanci TO zamestnanci_nov;
```

## 9. Vložení dat do tabulky
Data do tabulky se vkládají pomocí příkazu `INSERT INTO`.

```sql
INSERT INTO zamestnanci (id, jmeno, prijmeni, datum_narozeni)
VALUES (1, 'Jan', 'Novak', TO_DATE('1990-01-01', 'YYYY-MM-DD'));
```

```sql
INSERT INTO zamestnanci (id, jmeno, prijmeni, datum_narozeni)
VALUES (2, 'Petr', 'Kral', TO_DATE('1985-05-12', 'YYYY-MM-DD')),
       (3, 'Eva', 'Svobodova', TO_DATE('1992-08-30', 'YYYY-MM-DD'));
```

## 10. Aktualizace dat v tabulce
Pro změnu existujících dat v tabulce použijte příkaz UPDATE.

```sql
UPDATE zamestnanci
SET jmeno = 'Martin', prijmeni = 'Dvorak'
WHERE id = 1;
```

## 11. Smazání dat z tabulky
Data z tabulky se maže pomocí příkazu DELETE.

```sql
DELETE FROM zamestnanci
WHERE id = 2;
```

```sql
DELETE FROM zamestnanci;
```

## 12. Ověření změn (commit a rollback)
Pokud používáte transakce a chcete potvrdit nebo zrušit změny, použijte příkazy COMMIT a ROLLBACK.

COMMIT: Potvrzuje změny v databázi.

```sql
COMMIT;
```


```sql
ROLLBACK;
```

## 13. Funkce pro práci s řetězci

```sql
SELECT CONCAT('Jmeno: ', jmeno) FROM zamestnanci;
```

```sql
SELECT SUBSTR(jmeno, 1, 3) FROM zamestnanci; -- První 3 znaky
```

```sql
SELECT LENGTH(jmeno) FROM zamestnanci;

```

```sql
SELECT TRIM(' ' FROM jmeno) FROM zamestnanci;
```

```sql
SELECT REPLACE(jmeno, 'Jan', 'Petr') FROM zamestnanci;
```

```sql
SELECT TO_DATE('2025-03-19', 'YYYY-MM-DD') FROM dual;
```

```sql
SELECT TRIM(' ' FROM jmeno) FROM zamestnanci;
```