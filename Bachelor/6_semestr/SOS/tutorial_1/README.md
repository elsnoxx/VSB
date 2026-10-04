# Task 01

## Zadání

Nainstalujte do virtualizovaného stroje Debian (GNU/Debian) v minimalistickém režimu — bez grafického prostředí.

Požadavky:
- Přidejte dvě síťové karty: jednu do sítě NAT a druhou do Host-Only.
- Při instalaci vytvořte uživatele se stejným loginem, jaký používáte ve školním systému.
- Ověřte, že funguje přihlášení přes SSH.

Odevzdání:
- Do systému `kelvin` odevzdejte textový soubor obsahující výpis výstupů příkazů `ls -la /` a `ip add`.

Poznámka: Výsledek není nutné prezentovat vyučujícímu.

## Postup (stručně)
1. Spusťte instalaci Debianu a vyberte minimalistickou instalaci (bez GUI).
2. Přidejte dvě síťová rozhraní (NAT + Host-Only).
3. Vytvořte uživatele s vaším školním loginem.
4. Nakonfigurujte SSH a otestujte připojení z hostitele.
5. Vygenerujte textový soubor s výpisy příkazů a odešlete ho do `kelvin`.

## Kontrola SSH
Ověřte, že se můžete přihlásit přes SSH například takto (z hostitele):

```bash
ssh skolni_login@IP_ADRESA_VM
```

## Problémy s apt
Pokud narazíte na problémy s `apt`, zkontrolujte obsah `/etc/apt/sources.list`:

```bash
cat /etc/apt/sources.list
```

Pokud instalátor přidal záznamy s CD-ROM (`deb cdrom:`), doporučuji je zakomentovat (přidat `#` na začátek řádku). Například:

```bash
# deb cdrom:[Debian GNU/Linux 13.3.0 _Trixie_ - Official amd64 DVD Binary-1 with firmware 20260110-11:00]/ trixie contrib main non-free-firmware
```

Po úpravě spusťte aktualizaci:

```bash
sudo apt update
```

---

V případě, že chcete, mohu provést jemné stylistické úpravy textu, přidat kontrolní checklist nebo vytvořit šablonu pro odevzdání.