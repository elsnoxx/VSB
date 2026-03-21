#!/bin/bash

# --- Proměnné (uprav podle svého výstupu 'ip addr') ---
IF_NAT="enp0s3"       # Rozhraní s internetem (NAT)
IF_HOSTONLY="enp0s8"  # Rozhraní pro komunikaci s hostitelem

# 1. Vyčištění stávajících pravidel
iptables -F
iptables -X
iptables -t nat -F
iptables -t nat -X

# 2. Nastavení výchozí politiky (vše zakázat)
iptables -P INPUT DROP
iptables -P OUTPUT DROP
iptables -P FORWARD DROP

# 3. Povolení Loopbacku (vnitřní komunikace systému)
iptables -A INPUT -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT

# 4. SSH (22) a NFS (2049) - Pouze přes Host-only rozhraní
# Povolujeme TCP pro obojí, NFS vyžaduje i UDP
iptables -A INPUT -i $IF_HOSTONLY -p tcp -m multiport --dports 22,2049 -j ACCEPT
iptables -A OUTPUT -o $IF_HOSTONLY -p tcp -m multiport --sports 22,2049 -j ACCEPT
iptables -A INPUT -i $IF_HOSTONLY -p udp --dport 2049 -j ACCEPT
iptables -A OUTPUT -o $IF_HOSTONLY -p udp --sport 2049 -j ACCEPT

# 5. HTTP (80) a HTTPS (443) - Ze všech rozhraní
iptables -A INPUT -p tcp -m multiport --dports 80,443 -j ACCEPT
iptables -A OUTPUT -p tcp -m multiport --sports 80,443 -j ACCEPT

# 6. ICMP (Ping) - Na všech rozhraních
iptables -A INPUT -p icmp -j ACCEPT
iptables -A OUTPUT -p icmp -j ACCEPT

# 7. NAT (Source NAT / Masquerade) na NAT rozhraní
# Aby vnitřní síť mohla ven pod IP adresou rozhraní $IF_NAT
iptables -t nat -A POSTROUTING -o $IF_NAT -j MASQUERADE

# 8. Povolení navázaných spojení (DŮLEŽITÉ pro funkčnost odpovědí)
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

echo "Firewall byl úspěšně nastaven."