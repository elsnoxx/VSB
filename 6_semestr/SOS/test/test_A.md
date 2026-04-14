# Zadani A

miising tas 3 and 4 a vyzkouset to na konfigurovat na virtualu

## Task 1 - Instalace Debianu

1. Vytvorim novy VM, pojmenuji a bez pridani iso kliknu dokoncit
2. Pridat sitovy adapter typu **Síť pouze s hostem**
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
apt install vim isc-dhcp-client iptables nfs-kernel-server nfs-common -y
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
vim /etc/network/interfaces 
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


## Ukol 4
Vytvořte spustitelný skript v jazyce bash, který do systému přidá definovaný počet uživatelských účtů ve tvaru uz001 až uz###, kde ### bude číslo zadané jako parametr skriptu. Zařiďte, aby číslo nemohlo přesáhnout 3 cifry. Interpret pro všechny uživatele bude /bin/bash a uživatelům se vytvoří domovský adresář ve složce /home. Uživatelé budou mít prázdné heslo a budou nuceni si ho po prvním přihlášení změnit. Každému uživateli se při vytvoření účtu vytvoří v domovské složce soubor READ_ME.txt . Všem uživatelů definujte diskové kvóty.

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
for i in $(seq 1 10); do
    useradd -m -G projekty uzivatel$i
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

Prihlasit se jako uzivatel

```shell
fic0024@debian:~$ mkdir public_html
fic0024@debian:~$ echo "<?php phpinfo(); ?>" > ~/public_html/index.php
fic0024@debian:~$ chmod 755 /home/fic0024
fic0024@debian:~$ chmod 755 /home/fic0024/public_html
fic0024@debian:~$ chmod 644 /home/fic0024/public_html/index.php
```

Nastaveni PHP

```shell
nano /etc/apache2/mods-enabled/dir.conf
```

zmint na 
```
DirectoryIndex index.php index.cgi index.pl index.html index.xhtml index.htm
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

### Vytoreni sluzby
```shell
nano /etc/systemd/system/fwtest.service
```

```text
[Unit]
Description=Firewall sluzba

[Service]
Type=oneshot
WorkingDirectory=/root
ExecStart=/root/firewall.sh
ExecStop=/root/firewall_stop.sh
RemainAfterExit=yes
syste
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

### start skript

```text
#!/bin/bash

iptables -P INPUT DROP;
iptables -P OUTPUT DROP;

iptables -A INPUT  -i enp0s8 -p tcp --dport 22 -j ACCEPT
iptables -A INPUT  -i enp0s8 -p tcp --dport 2049 -j ACCEPT
iptables -A INPUT  -i enp0s8 -p udp --dport 2049 -j ACCEPT

iptables -A INPUT  -p tcp -m multiport --dports 80,443 -j ACCEPT

iptables -A INPUT -p icmp -j ACCEPT

iptables -A OUTPUT -p icmp -j ACCEPT
```

### End skript

```test
#!/bin/bash

iptables -F
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

Zmenit
```text
INTERFACESv4="enp0s8"
```

### konfigurace DHCP serveru

```shell
nano /etc/dhcp/dhcpd.conf
```

```text
> option domain-name "vsb.cz";
  option domain-name-servers 158.196.0.53, 158.196.149.9;
> subnet 192.168.56.0 netmask 255.255.255.0 {
  range 192.168.56.20 192.168.97.120;
  option broadcast-address 192.168.56.255;
  option routers 192.168.56.2;
}
```

Kontroloa servis
```shell
service isc-dhcp-server restart

service isc-dhcp-server status

dhcp-lease-list
```

# Task 9 - NFS
Nainstalujte server NFS a vyexportujte složku /var/www/html pro všechny počítače v síti „Host-only network“. Správnou funkci demonstrujte.

```shell
nano /etc/exports
```

```text
/var/www        192.168.97.0/24(rw,sync,no_subtree_check)
```

Restarttovat servisu
```shell
service nfs-kernel-server restart

exportfs
dhcpd -d # pro ziskani IP z meho DHCP
nano /etc/dhcp/dhcpd.conf
```

## Testovani v konzoli druheho PC
apt install nfs-common
mount 192.168.97.102:/var/www /mnt/