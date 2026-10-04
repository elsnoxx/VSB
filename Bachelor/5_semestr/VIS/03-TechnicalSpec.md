# Technická specifikace

## Doménový model

==:> Todo

## Odhady velikosti

| Jméno (Entita) | Odhadovaný Počet (1. rok) | Velikost jedné položky (Bytes) | Velikost všech (MB) |
|---|---|---|---|
| Zařízení | 10000 | 550 B | 5.25 MB |
| Historie Pohybu/Stavů | 30000 | 80 B | 2.29 MB |
| Uživatel | 100 | 600 B | 0.06 MB |
| Lokace | 500 | 300 B | 0.14 MB |
| Typ Zařízení | 50 | 250 B | 0.01 MB |
| Stav Zařízení | 20 | 200 B | 0.00 MB |
| Role a Oprávnění | 10 | 150 B | 0.00 MB |
| **CELKEM ZA ROK** | | | **cca 7.75 MB** |

## Uložení dat

Všechna data budou uložena v databázi MariaDB.

## Interakce Aktérů
- Operátor: Přístup přes webový prohlížeč, může vyhledávat, procházet a zobrazovat detaily.
- Administrátor: Kompletní správa systémových dat (rolí, typů, stavů, lokací).
- IT Technik: Základní přístup – prohlížení seznamu, vyhledávání detailů zařízení a přihlášení.


## Použité technologie
Zde uvedete všechny klíčové technologie, které jste definoval v sekci "Jak?" ve vizi projektu.

- Databáze: MariaDB
- Backend: ASP.NET Core (pro serverovou logiku a REST API)
- Frontend: React framework (pro uživatelské rozhraní)
- Hosting: Linuxový server (Docker Compose)
- Verzování kódu: Git + Github