# Zadani A

miising tas 3 and 4 a vyzkouset to na konfigurovat na virtualu

## Task 1 - Instalace Debianu

1. Vytvorim novy VM, pojmenuji a bez pridani iso kliknu dokoncit
2. Pridat sitovy adapter typu **Síť pouze s hostem**
3. kontrola jestli u adapteru je zapnute dhcp
4. Kliknu na Uloziste
5. Najdu radic sata tam je ikona hardisku **add hardisk**
6. Create
7. dokoncit 3x
3. Po startu VM vyplnim iso disk a kliknu na mount a retry boot
4. Po menu vybrat moznost instal
5. Vybrat jazyk jako english
6. Umistneni potom others, potom europe a tam je k nalezeni czech
7. Vybrat unitstates
8. Keymap: american english
9. Vybrat enp0s3
10. hostname nechat
11. domenu nechat prazdnou
12. heslo pro root -- root 
13. user fic0024 - fic
14. Guided - use entire disk
15. Vybrat dany disk
16. Vybrat All in one partion
17. Vybrat Finish parttioning and write change to disk
18. Yes
19. No - neskenovat dalsi media
20. Yes - netwrok mirror
21. Czechia
22. ftp.debian.org
23. nechat prazden a continue
24. No - pruzkum popularity
25. Odebrat vse pomoci mezery, vybrat jen SSH server a potom enter pro potvrzeni
26. Yes - GRUB boot loader
27. vybrat /dev/sda
28. Reboot

### Opraveni balickovaci sluzby

```shell
nano /etc/apt/sources.list
```

Prvni zakomentovat a potom ulozit

### instalovani vseh potrebnych balicku
```bash
apt install vim isc-dhcp-client isc-dhcp-server mdadm quota iptables nfs-kernel-server nfs-common curl -y
```

## Ukol 2 Konfigurace site
Nakonfigurujte systém tak, aby síťová karta na rozhraní NAT dostávala IP adresu prostřednictvím protokolu DHCP (z Virtualboxu) a druhá karta bude mít IP adresu nastavenou staticky. Pro konfiguraci obou rozhraní využijte standardní metody používané v distribuci Debian.



2. Najit a nastativ

```bash
ip a
dhclient enp0s8
```

3. Automaticky nastaveni

```bash
nano /etc/network/interfaces 
```

```text
# This file describes the network interfaces available on your system
# and how to activate them. For more information, see interfaces(5).

source /etc/network/interfaces.d/*

# The loopback network interface
auto lo
iface lo inet loopback

# The primary network interface
allow-hotplug enp0s3
iface enp0s3 inet dhcp

# Secondery network interface
allow-hotplug enp0s8
iface enp0s8 inet static
        address 192.168.56.104
        netmask 255.255.255.0

# This is an autoconfigured IPv6 interface
iface enp0s3 inet6 a
```

Kontrola nastaveni a pri startu bude aktivni ip u interfaceu
```bash
reboot
```

## Ukol 3
Do virtualizovaného PC přidejte další tři pevné disky o kapacitě alespoň 200MB. Z těchto disků vytvořte v systému RAID který bude odolný proti výpadku dvou disků. Na RAID vytvořte jeden oddíl a naformátujte ho souborovým systémem ext4. Tento souborový systém připojte jako složku /home. Nakonfigurujte systém tak, aby připojení diskového pole proběhlo vždy po startu systému, pro identifikaci raidu použijte UUID.

Nejsprve zacnu prikazem abych vedel nazyvani disku
```shell
lsblk
```

Potom vytvorit partitiony
```shell
fdisk /dev/sda
fdisk /dev/sdb
fdisk /dev/sdc
```
postup pro vsechny
1. n (nový) -> p (primární) -> Enter -> Enter -> Enter (vše výchozí).
2. t (změna typu) -> fd (Linux raid autodetect).
3. w (uložit a skončit).



```shell
mdadm --create /dev/md0 --level=1 --raid-devices=3 /dev/sda1 /dev/sdb1 /dev/sdc1
```
potom dam y


formatovani disku
```shell
mkfs.ext4 /dev/md0
```

### Mount a přesun dat:
```shell
Bash
mount /dev/md0 /mnt
cp -rp /home/* /mnt/
umount /mnt
```

UUID a fstab:

Zjisti UUID: blkid /dev/md0
V /etc/fstab přidej: UUID=5b0f21c6-77e0-4c10-bf2a-018e7dafcc86 /home ext4 defaults,usrquota 0 2

### tetovaci vypisy

df -h /home

cat /proc/mdstat
ls -l /home

## Ukol 4
Vytvořte spustitelný skript v jazyce bash, který do systému přidá definovaný počet uživatelských účtů ve tvaru uz001 až uz###, kde ### bude číslo zadané jako parametr skriptu. Zařiďte, aby číslo nemohlo přesáhnout 3 cifry. Interpret pro všechny uživatele bude /bin/bash a uživatelům se vytvoří domovský adresář ve složce /home. Uživatelé budou mít prázdné heslo a budou nuceni si ho po prvním přihlášení změnit. Každému uživateli se při vytvoření účtu vytvoří v domovské složce soubor READ_ME.txt . Všem uživatelů definujte diskové kvóty.

### Script 

```bash
#!/bin/bash

# 1. Kontrola, zda byl zadán parametr
if [ -z "$1" ]; then
    echo "Použití: $0 <počet_uživatelů>"
    exit 1
fi

# 2. Kontrola, zda je parametr číslo a nepřesahuje 999
pocet=$1
if ! [[ "$pocet" =~ ^[0-9]+$ ]] || [ "$pocet" -gt 999 ]; then
    echo "Chyba: Zadejte číslo v rozsahu 1 až 999."
    exit 1
fi

echo "Vytvářím $pocet uživatelů..."

# 3. Cyklus pro vytváření uživatelů
for i in $(seq -f "%03g" 1 "$pocet"); do
    username="uz$i"

    # Vytvoření uživatele (-m vytvoří home, -s nastaví shell)
    useradd -m -s /bin/bash "$username"

    # Nastavení prázdného hesla
    passwd -d "$username"

    # Vynucení změny hesla při prvním přihlášení
    chage -d 0 "$username"

    # Vytvoření souboru READ_ME.txt v domovské složce
    echo "Vítejte v systému, prosím změňte si heslo." > "/home/$username/READ_ME.txt"
    
    # Nastavení vlastnictví souboru uživateli
    chown "$username:$username" "/home/$username/READ_ME.txt"

    # Nastavení diskové kvóty (příklad: 100MB soft, 110MB hard)
    # Předpokládá se, že kvóty jsou na systému již inicializovány
    setquota -u "$username" 100M 110M 0 0 /home

    echo "Uživatel $username vytvořen."
done

echo "Hotovo."
```

## Ukol 5
V kořenovém adresáři vytvořte složku /projekty. V systému vytvořte skupinu projekty a přidejte do ní deset uživatelů. Složka /projekty bude umožňovat přístup (rwx) jen uživatelům patřícím do skupiny projekty. Pokud některý z uživatelů vytvoří v této složce soubor, tento bude automaticky patřit skupině projekty a nikoli domovské skupině uživatele, který ho vytvořil.

### vytvoreni slozky

```shell
mkdir /projekty	
```

### Pridani skupiny

```shell
addgroup projekty
```

### Vytvoreni uzivatelu a pridani do skupiny

```shell
adduser user1
adduser user2
```

```shell
usermod -a -G projekty user1
usermod -a -G projekty user2
```

pro 10 uzivatelu

```bash
for i in {3..10}; do
    adduser --disabled-password --gecos "" user$i
    usermod -a -G projekty user$i
done
```

### zmena prav na slozku

```shell
chgrp projekty /projekty #vlastnikem slozky je skupina
chown root:projekty /projekty 
chmod -R 2770 /projekty #rekurzivni zmena prav (vlastnikem je skupina)
```

### vypis informaci
```shell
cat /etc/passwd
cat /etc/group

ls -ld /projekty
```

### kontrola uzivatelem
```shell
su - user1
touch /projekty/test_soubor
ls -l /projekty/test_soubor
```

## Ukol 6
Nainstalujte webový server Apache2 s podporou PHP, https a userdir.

Instalace Apache + PHP

```bash
apt install apache2 php libapache2-mod-php php-mcrypt -y
systemctl status apache2
```

**URL to run:** [http://192.168.56.104/](http://192.168.56.104/)

Nastaveni SSL

```bash
a2enmod ssl
nano /etc/apache2/sites-available/000-default.conf
```

nebo 

```shell
a2enmod ssl
a2ensite default-ssl
```

```text
<VirtualHost *:443>
        # The ServerName directive sets the request scheme, hostname and port that
        # the server uses to identify itself. This is used when creating
        # redirection URLs. In the context of virtual hosts, the ServerName
        # specifies what hostname must appear in the request's Host: header to
        # match this virtual host. For the default virtual host (this file) this
        # value is not decisive as it is used as a last resort host regardless.
        # However, you must set it for any further virtual host explicitly.
        #ServerName www.example.com

        ServerAdmin webmaster@localhost
        DocumentRoot /var/www/html

        SSLEngine on
        SSLCertificateFile /etc/ssl/certs/apache-selfsigned.crt
        SSLCertificateKeyFile /etc/ssl/private/apache-selfsigned.key

        # Available loglevels: trace8, ..., trace1, debug, info, notice, warn,
        # error, crit, alert, emerg.
        # It is also possible to configure the loglevel for particular
        # modules, e.g.
        #LogLevel info ssl:warn

        ErrorLog ${APACHE_LOG_DIR}/error.log
        CustomLog ${APACHE_LOG_DIR}/access.log combined

        # For most configuration files from conf-available/, which are
        # enabled or disabled at a global level, it is possible to
        # include a line for only one particular virtual host. For example the
        # following line enables the CGI configuration for this host only
        # after it has been globally disabled with "a2disconf".
        #Include conf-available/serve-cgi-bin.conf
</VirtualHost>
```

```bash
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout /etc/ssl/private/apache-selfsigned.key -out /etc/ssl/certs/apache-selfsigned.crt

systemctl restart apache2
```

Instalace userdir

```shell
a2enmod userdir
systemctl restart apache2
```

Nastaveni PHP

```shell
nano /etc/apache2/mods-enabled/dir.conf
```

```shell
nano /etc/apache2/mods-available/php8.4.conf
```

```text
<IfModule mod_userdir.c>
    <Directory /home/*/public_html>
        php_admin_flag engine On
    </Directory>
</IfModule>
```

Prihlasit se jako uzivatel

```shell
fic0024@debian:~$ mkdir public_html
fic0024@debian:~$ echo "<?php phpinfo(); ?>" > ~/public_html/index.php
fic0024@debian:~$ chmod 755 /home/fic0024
fic0024@debian:~$ chmod 755 /home/fic0024/public_html
fic0024@debian:~$ chmod 644 /home/fic0024/public_html/index.php
```



zmint na 
```
DirectoryIndex index.php index.cgi index.pl index.html index.xhtml index.htm
```


potom restart servisy
```shell
systemctl restart apache2
```

## Ukol 7

7) Vytvořte skript, který nastaví firewall systému tak, aby defaultní politika pro INPUT a OUTPUT na všech rozhraních byla DROP.

Povolte do systému přístup (INPUT, OUTPUT) pro službu SSH (port 22 tcp)a NFS (port 2049 tcp,udp) pouze prostřednictvím rozhraní „Host-only network“.
Povolte přístup (INPUT, OUTPUT) na http a https ze všech rozhraní a IP adres.
Povolte provoz ICMP (INPUT, OUTPUT) na všech rozhraních.
Na rozhraní připojeném k síti NAT nakonfigurujte překlad síťových adres (source nat).
Funkčnost firewallu demonstrujte.

Vytvořte jednotku pro systemd, které zavede pravidla firewalu, vždy po startu systému.

### start skript

```shell
nano firewall.sh
```

```text
#!/bin/bash

# 1. Čištění starých pravidel
iptables -F
iptables -t nat -F

# 2. Nastavení defaultní politiky na DROP
iptables -P INPUT DROP
iptables -P OUTPUT DROP
iptables -P FORWARD DROP

# 3. Povolení loopbacku (důležité pro systémové služby)
iptables -A INPUT -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT

# 4. Povolení navázaných spojení (ESTABLISHED, RELATED) - KLÍČOVÉ PRO SSH
# Toto dovolí serveru odpovídat na požadavky, které byly povoleny v INPUTu
iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
iptables -A OUTPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT

# 5. SSH (22) a NFS (2049) pouze na Host-only (enp0s8)
iptables -A INPUT -i enp0s8 -p tcp --dport 22 -j ACCEPT
iptables -A OUTPUT -o enp0s8 -p tcp --sport 22 -j ACCEPT

iptables -A INPUT -i enp0s8 -p tcp --dport 2049 -j ACCEPT
iptables -A OUTPUT -o enp0s8 -p tcp --sport 2049 -j ACCEPT
iptables -A INPUT -i enp0s8 -p udp --dport 2049 -j ACCEPT
iptables -A OUTPUT -o enp0s8 -p udp --sport 2049 -j ACCEPT

# 6. HTTP (80) a HTTPS (443) ze všech rozhraní
iptables -A INPUT -p tcp -m multiport --dports 80,443 -j ACCEPT
iptables -A OUTPUT -p tcp -m multiport --sports 80,443 -j ACCEPT

# 7. ICMP (ping) na všech rozhraních
iptables -A INPUT -p icmp -j ACCEPT
iptables -A OUTPUT -p icmp -j ACCEPT

# 8. Source NAT na rozhraní NAT (pravděpodobně enp0s3)
# Toto zajistí, že virtuál může "ven" do internetu přes rozhraní enp0s3
iptables -t nat -A POSTROUTING -o enp0s3 -j MASQUERADE
```

### End skript
```shell
nano firewall_stop.sh
```

```test
#!/bin/bash

# 1. Nastav politiku na přijímání (vše projde)
iptables -P INPUT ACCEPT
iptables -P OUTPUT ACCEPT
iptables -P FORWARD ACCEPT

# 2. Teď teprve smaž pravidla
iptables -F
iptables -t nat -F
```

### povoleni execute
```shell
chmod +x firewall*
```

### Vytoreni sluzby
```shell
nano /etc/systemd/system/fwtest.service
```

```text
[Unit]
Description=Firewall sluzba
After=network.target

[Service]
Type=oneshot
ExecStart=/root/firewall.sh
ExecStop=/root/firewall_stop.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

```shell
systemctl daemon-reload
systemctl enable fwtest
systemctl start fwtest
```

### Pravidla pro FW

```shell
iptables -L #zobrazi pravidla
iptables -F #smaze pravidla
```



### Test
```text
iptables -L -v -n

ssh fic0024@192.168.56.104
ssh fic0024@10.0.2.15

ping -c 4 8.8.8.8

curl -I http://google.com
curl -I http://localhost
```

## Ukol 8 - DHCP server
Nainstalujte server DHCP a nakonfigurujte ho tak, aby v síti Host only network dynamicky přiřazoval 100 ipv4 adres. Spolu s IP adresou předávejte klientským počítačům také všechna potřebná nastavení pro jejich práci v síti.

Otevrit konfiguraci dhcp serveru

```shell
nano /etc/default/isc-dhcp-server
```

Pridat
```text
INTERFACESv4="enp0s8"
```

### konfigurace DHCP serveru

```shell
nano /etc/dhcp/dhcpd.conf
```

```text
option domain-name "vsb.cz";
option domain-name-servers 158.196.0.53, 158.196.149.9;

subnet 192.168.56.0 netmask 255.255.255.0 {
  range 192.168.56.20 192.168.56.120;
  option broadcast-address 192.168.56.255;
  option routers 192.168.56.2;
}
```

Kontroloa servis
```shell
service isc-dhcp-server restart

service isc-dhcp-server status

dhcp-lease-list

dhcpd -t -cf /etc/dhcp/dhcpd.conf
```

### testovani

zapnou novu cistou virtualku na dhcp serveru dam prikaz 
```shell
journalctl -u isc-dhcp-server -f
```

a pripojim k adapteru na novem virtualu ip astresu jourmnal by mel potom vypsat pripojeni


# Task 9 - NFS
Nainstalujte server NFS a vyexportujte složku /var/www/html pro všechny počítače v síti „Host-only network“. Správnou funkci demonstrujte.

```shell
nano /etc/exports
```

```text
/var/www    192.168.245.0/24(rw,sync,no_subtree_check,no_root_squash)
```

Restarttovat servisu
```shell
# Znovu načte konfiguraci exportů
exportfs -ra

# Restartuje službu pro jistotu
service nfs-kernel-server restart

# Ověří, že je složka viditelná pro správnou síť
exportfs -v

service nfs-kernel-server restart

exportfs
dhcpd -d # pro ziskani IP z meho DHCP
nano /etc/dhcp/dhcpd.conf
```

## Testovani v konzoli druheho PC
```shell
apt install nfs-common -y
dhclient enp0s8
mkdir -p /mnt/web_nfs
mount 192.168.245.104:/var/www /mnt/web_nfs
```

vytvoreni testovaciho souboru
```shell
# Na klientovi
touch /mnt/web_nfs/html/funguje_to.txt

# Na serveru
ls -l /var/www/html/
```


### aplikovani zmen
```shell
exportfs -ra
```