mkdir /projekty				#nova slozka pro sdileni
addgroup projekty 		#nova skupina se jmenem 

adduser user1
adduser user2

usermod -a -G projekty user1	#pridani uzivatele do skupiny
usermod -a -G projekty user2
...............
chgrp projekty /projekty #vlastnikem slozky je skupina
chmod -R 2770 /projekty #rekurzivni zmena prav (vlastnikem je skupina)


#informace o userech
cat /etc/passwd
cat /etc/group

ls -ld prava adresare
