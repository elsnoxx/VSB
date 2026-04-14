

#Tvorba sluzby
nano /etc/systemd/system/fwtest.service

-----------------------------------
[Unit]
Description=Firewall sluzba

[Service]
Type=oneshot
WorkingDirectory=/root
ExecStart=/root/firewall.sh				#Spousteci skript
ExecStop=/root/firewall_stop.sh			#Vypinaci skript
RemainAfterExit=yes						#Po dokonceni ponechat sluzbu aktivni
syste
[Install]
WantedBy=multi-user.target
----------------------------------

#Zapnuti sluzby
systemctl enable fwtest


apt update
apt install iptables

iptables -L #zobrazi pravidla
iptables -F #smaze pravidla


#Skript pro start firewall.sh	
-----------------------------------
#!/bin/bash

iptables -P INPUT DROP;
iptables -P OUTPUT DROP;

iptables -A INPUT  -i enp0s8 -p tcp --dport 22 -j ACCEPT
iptables -A INPUT  -i enp0s8 -p tcp --dport 2049 -j ACCEPT
iptables -A INPUT  -i enp0s8 -p udp --dport 2049 -j ACCEPT

iptables -A INPUT  -p tcp -m multiport --dports 80,443 -j ACCEPT

iptables -A INPUT -p icmp -j ACCEPT

iptables -A OUTPUT -p icmp -j ACCEPT



-----------------------------------

#Skript pro stop firewall_stop.sh	
-----------------------------------
#!/bin/bash

iptables -F
-----------------------------------
