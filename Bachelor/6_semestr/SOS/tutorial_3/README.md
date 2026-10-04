# Uživatelé, práva, quoty, iptables

## Uživatelé

Vytvořte v systému 50 uživatelů s prázdným heslem a donuťte uživatele ke změně hesla při prvním přihlášení.
Do systému kelvin odevzdejte výpis /etc/passwd a /etc/shadow
Vytvořte skupinu "tisk" a složku /tisk. Práva složky nastavte na všechny pro tuto skupinu. Zařiďte, aby soubory vznikající v této složce patřily skupině tisk a né domovské skupině uživatele, který soubor vytvořil. Do skupiny tisk přidejte tři náhodné uživatele.
Do systému kelvin odevzdejte výpis práv složky /test a výpis cat /etc/group | grep tisk.

## Quoty

Přidejte do systému další virtualizovaný pevný disk a přesuňte na něj složku /home. Nad složkou /home zprovozněte diskové quoty pro uživatele. Správně proveďte úpravy v souboru /etc/fstab.
Do systému kelvin odevzdejte výpis programu repquota -a.
Pozn: Na tutoriálu jsme si neukazovali možnost kopírování quot od nějakého uživatele. Dělá se to pomocí edquota -p uživatel-prototyp jiny-uživatel.

## IP Tables

Vytvořte skript který zavede následující pravidal firewalu: - Povolte do systému přístup (INPUT, OUTPUT) pro službu SSH (port 22 tcp)a NFS (port 2049 tcp,udp) pouze prostřednictvím rozhraní „Host-only network“. - Povolte přístup (INPUT, OUTPUT) na http a https ze všech rozhraní a IP adres. - Povolte provoz ICMP (INPUT, OUTPUT) na všech rozhraních. - Na rozhraní připojeném k síti NAT nakonfigurujte překlad síťových adres (source nat).

Firewall otestujte pomocí utilitky nc případně pomocí pingu.
Do systému kelvin odevzdejte vámi vytvořený skript.