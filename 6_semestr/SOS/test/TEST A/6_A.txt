apt update 
apt install apache2 php libapache2-mod-php php-mcrypt

a2enmod ssl  #zapnuti ssl

/etc/apache2/sites-available/000-default.conf  #nastaveni ssl

# Instalace weboveho serveru Apache s provozem http a modulem userdir
apt update
apt install libapache2-mod-php
a2enmod userdir
systemctl restart apache2
## v druhe konzoli
mkdir public_html
cp /var/www/html/index.html ./public_html/

#Pokud chceme vlozit i soubor s ukazkou funknosti.

#Nastaveni a restart serveru
ls /etc/apache2/mods-enabled

/etc/apache2/mods-enabled/dir.conf 			#prehodit poradi index.php pred index.html

#Vytvorit soubor index.php
/var/www/html/index.php
