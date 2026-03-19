#!/bin/bash
# Cesta k souboru, který chceme hlídat
LOGFILE="/tmp/cron_test.log"

# Získání velikosti logu v kB
# Pokud soubor neexistuje, du vyhodí chybu, proto přesměrujeme chybu do /dev/null
size=$(du -s "$LOGFILE" 2>/dev/null | cut -f1)

# Pokud velikost přesáhne 100 kB, soubor smažeme
# -n kontroluje, zda je proměnná size neprázdná (pro případ, že soubor ještě neexistuje)
if [ -n "$size" ] && [ "$size" -gt 100 ]; then
    rm "$LOGFILE"
fi

# Zápis do logu
echo "Skript byl spuštěn v: $(date)" >> "$LOGFILE"
