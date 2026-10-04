# zadani

## Systemd

Prosím vyberte si jedno zadání a) ze čtvrté přednášky a vytvořte jednoduchý skript, který bude po startuspouštěn automaticky. O zavedení skriptu po startu systému se postará SYSTEMD pro který korektně vytvořte a zaveďte jednotku.

Do systému kelvin odenzdejte: skript v jazyce BASH, konfigurační soubor jednotky a výpis programu systemctl status moje_jednotka spuštěného po startu systému.

## Crontab

Vytvořte libovolný triviální skript v jazyce BASH, který bude cyklicky spouštěný pomocí systémového CRONU. Pro cyklické spouštění skriptu vytvořte konfigurační soubor ve složce /etc/cron.d/

Do systému kelvin odevzdejte: skript v jazyce BASH, konfigurační soubor pro CRON

## Disky, RAID

Prosím vemte číslo vašeho loginu a proveďte %5 + 1, takové zadání vypracujte. Jako druhou část zadání se prosím pokuste váš RAID rozbít (označit disk jako vadný) a zase opravit.

1.zadání
Do svého virtualizovaného PC přidejte další SATA disk. Tento disk naformátujte souborovým systémem ext4 a připojte ho jako adresář /home. Přičemž zachovejte veškerý obsah původního adresáře /home. Do svého virtualizovaného PC přidejte další tři SATA disky. Vytvořte nad nima SW RAID1 s jedním spare diskem.

2.zadání
Do svého virtualizovaného PC přidejte další SATA disk. Tento disk naformátujte souborovým systémem ext4 a připojte ho jako adresář /home. Přičemž zachovejte veškerý obsah původního adresáře /home. Do svého virtualizovaného PC přidejte další tři SATA disky. Vytvořte nad nima SW RAID5 bez použití spare disku.

3.zadání
Do svého virtualizovaného PC přidejte další SATA disk. Tento disk naformátujte souborovým systémem ext4 a připojte ho jako adresář /home. Přičemž zachovejte veškerý obsah původního adresáře /home. Do svého virtualizovaného PC přidejte další čtyři SATA disky. Vytvořte nad nima SW RAID5 s jedním spare diskem.

4.zadání
Do svého virtualizovaného PC přidejte další SATA disk. Tento disk naformátujte souborovým systémem ext4 a připojte ho jako adresář /home. Přičemž zachovejte veškerý obsah původního adresáře /home. Do svého virtualizovaného PC přidejte další čtyři SATA disky. Vytvořte nad nima SW RAID6 bez použití spare disku.

5.zadání
Do svého virtualizovaného PC přidejte další SATA disk. Tento disk naformátujte souborovým systémem ext4 a připojte ho jako adresář /home. Přičemž zachovejte veškerý obsah původního adresáře /home. Do svého virtualizovaného PC přidejte dalších pět SATA disků. Vytvořte nad nima SW RAID6 s jedním spare diskem.

Do systému kelvin odevzdejte:
Výpis programu mount před a po dodání pevného disku a jeho připojení do složky home. Výpis stavu vašeho RAID pole před a po "havárii".