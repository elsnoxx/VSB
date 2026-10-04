apt install isc-dhcp-server

mcedit /etc/default/isc-dhcp-server

INTERFACESv4="enp0s8"

mcedit /etc/dhcp/dhcpd.conf

nano /etc/dhcp/dhcpd.conf
> option domain-name "vsb.cz";
  option domain-name-servers 158.196.0.53, 158.196.149.9;
> subnet 192.168.56.0 netmask 255.255.255.0 {
  range 192.168.56.20 192.168.97.120;
  option broadcast-address 192.168.56.255;
  option routers 192.168.56.2;
}


service isc-dhcp-server restart¨

service isc-dhcp-server status

dhcp-lease-list


zapnout druhy pocitac 
dhclient enp0s8
dhcpd -d