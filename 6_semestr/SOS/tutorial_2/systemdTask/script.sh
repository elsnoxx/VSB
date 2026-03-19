#!/bin/bash
# Cesta: /usr/local/bin/muj_skript.sh

# Procházíme všechny podadresáře v /home
for dir in /home/* ; do
    if [ -d "$dir" ]; then
        # Získání velikosti v kB (du -s vrátí velikost a název, cut vyřízne jen číslo)
        size=$(du -s "$dir" | cut -f1)

        # Pokud je velikost větší než 1 kB
        if [ "$size" -gt 1 ]; then
            touch "$dir/BYLA PREKROCENA MAX VELIKOST SLOZKY"
            # Volitelně nastavíme vlastníka, aby soubor viděl i uživatel
            chown --reference="$dir" "$dir/BYLA PREKROCENA MAX VELIKOST SLOZKY"
        fi
    fi
done



