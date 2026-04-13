# Instalace NFS serveru s exportem slozky /var/www
apt install nfs-kernel-server
nano /etc/exports
> /var/www        192.168.97.0/24(rw,sync,no_subtree_check)
service nfs-kernel-server restart
exportfs
dhcpd -d # pro ziskani IP z meho DHCP
nano /etc/dhcp/dhcpd.conf
## v konzoli druheho PC
apt install nfs-common
mount 192.168.97.102:/var/www /mnt/