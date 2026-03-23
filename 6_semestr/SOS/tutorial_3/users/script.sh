#!/bin/bash

# 1. Vytvoření uživatelů (pokud už existují, vynechá se)
for i in $(seq 1 50); do
    useradd -m "user$i" -p "" -s /bin/bash
    
    # Vynucení změny hesla
    chage -d 0 "user$i"
done

# 2. Skupina a složka
groupadd tisk 2>/dev/null

# Vytvoření složky (pokud neexistuje)
mkdir -p /tisk

# Nastavení vlastníka (root) a skupiny (tisk)
chown root:tisk /tisk

# Nastavení práv: 
chmod 2770 /tisk

# 3. Přidání 3 náhodných uživatelů do skupiny
usermod -aG tisk user1
usermod -aG tisk user10
usermod -aG tisk user20

ls -la /tisk

cat /etc/group | grep tisk >> tisk_group.txt

cat /etc/passwd > users.txt
cat /etc/shadow > shadow.txt