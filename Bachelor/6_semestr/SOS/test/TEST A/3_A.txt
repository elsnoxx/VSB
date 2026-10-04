#Zakladní postup
Vypnout virtuál
Pridat disky

#Prikazy 
fdisk -l 					#vypis disku
mcedit /etc/fstab
cat /proc/mdstat	
lsblk			

#Balicek na instalaci RAID
apt update			#aktualizace balicku
apt-get install mdadm		#instalace balicku

#Udelat partitiony pro kazdy disk
fdisk /dev/sdX      	#zapnuti nataveni disku
n ENTER					#new
p ENTER					#primary
ENTER, ENTER,.. 		#dalsi nastaveni
t ENTER fd ENTER		#nevim nejake nastaveni
w ENTER					#save nastaveni

#Tvorba RAID
mdadm --create /dev/md0 --level=1 --raid-devices=2 /dev/sdb /dev/sdc
            

#Alternativa pøímo se sparem
mdadm --create /dev/md0 --level=1 --raid-devices=2 /dev/sdb /dev/sdc --spare-devices=1 /dev/sdd	

#Partition
cfdisk /dev/md0
gpt
new
write

#Propojeni s /home/

mkfs.ext4 /dev/md0p1

mount /dev/md0p1 /mnt/

cp -a /home/* /mnt/

mount /dev/md0p1 /home/


#Co dat do /etc/fstab, aby se raid udelal i po spusteni
/dev/md127p1	/home	ext4	defaults	0	0


NEZAPOMENOUT, ZE SE PO RESTARTU MENI md0 na md127