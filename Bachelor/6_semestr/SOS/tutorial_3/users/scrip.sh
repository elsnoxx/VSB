#bin/bash

for i in $(seq 1 50); do
    # Vytvoření uživatele s prázdným heslem a domovskou složkou
    useradd -m "user$i" -p ""
    
    # Vynucení změny hesla při prvním přihlášení (expirace hesla)
    chage -d 0 "user$i"
done
