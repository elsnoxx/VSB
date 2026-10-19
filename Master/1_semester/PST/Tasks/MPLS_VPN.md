# Routers settings

## CE1
```bash
enable
configure terminal
hostname CE1

```

## CE2
```bash
enable
configure terminal
hostname CE2
```

## P
```bash
enable
configure terminal
hostname P
router ospf 1
network 10.0.10.0 0.0.0.255 area 0
network 10.0.11.0 0.0.0.255 area 0
network 10.0.2.1 0.0.0.0 area 0
mpls ldp router-id loopback0 force
interface fastethernet 0/0
mpls ip
interface serial 0/1/0
mpls ip
```

## PE1
```bash
enable
configure terminal
hostname PE1
router ospf 1
network 10.0.10.0 0.0.0.255 area 0
network 10.0.1.1 0.0.0.0 area 0
mpls ldp router-id loopback0 force
interface fastethernet 0/1
mpls ip
ip vrf ZAKAZNIK
rd 1:1
route-target export 1:1
route-target import 1:1
```

## PE2
```bash
enable
configure terminal
hostname PE2
router ospf 1
network 10.0.11.0 0.0.0.255 area 0
network 10.0.3.1 0.0.0.0 area 0
mpls ldp router-id loopback0 force
interface serial 0/1/0
mpls ip
ip vrf ZAKAZNIK
rd 1:1
route-target export 1:1
route-target import 1:1
```

## PCs
Pro získání dev rozhrani:
```bash
ip a
```

Pro zapnutí rozhraní:
```bash
sudo ip link set [device] up
```

### MPLS oblast OSPF area 0

```bash
sudo ip addr add 10.0.10.3/24 dev [device]
```

### EIGAEP AS 1 - levá

```bash
sudo ip addr add 192.168.0.1/24 dev [device]
```

### EIGAEP AS 1 - levá

```bash
sudo ip addr add 192.168.10.1/24 dev [device]
```