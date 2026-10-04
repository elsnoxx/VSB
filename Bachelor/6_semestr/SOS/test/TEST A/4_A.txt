#pred spustenim nejsou nastaveny quoty na home
apt install quota
mount -o remount,usrquota,grpquota /dev/md127p1
service quota start
quotaon /home

#editovani quoty
quota username
edquota username

edquota -p userprototype newuser; #upraveni quoty podle prototypu

#kde najdu info o uzivatelich
cat /etc/passwd

#je script spustitelny?
ls -l script

#spustit script
./script parametry

#Skript pro uzivatele
chmod +x skriptname.sh		#aby sel VSEMI spustit skript
https://pastebin.com/vBFKxXq8

-----------------------------------
#!/bin/bash

if ! [[ $1 =~ ^[0-9]+$ ]]
then
    echo "Neni cislo";
	exit;
fi;

if [ -z $1 ] || [ $1 -gt 999 ] || [ $1 -lt 1 ]; then
    echo "nulovy znak, vetsi nez 999, mensi nez 1";
	exit;
fi;

a=1
to=$1

prefix="uz"

while [[ a -le to ]]
do
   name="$prefix$a" 
   echo $name
	
   #-m specifikuje vytvoreni domovskeho adresare v /home
   #-s specifikuje shell
   #-c je comment (full name)
   #-p je password 
   
   useradd -m -s /bin/bash -c "$name" -p "" $name  
   #useradd -m -s /bin/bash -c "$name" -p `echo $1 | openssl passwd -1 -stdin` $name
   chage -d 0 "$name"				#nebo passwd -e user
   setquota $name 100 200 10 15 -a /home

   a=$(($a+1))
done
-----------------------------------

#Otestovani quota
repquota /dev/md127