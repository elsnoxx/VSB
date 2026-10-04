#!/bin/bash

echo "=== Kompilace FileGenerator ==="
cd FileGenerator
make clean
make
echo "Generator zkompilován úspěšně"


echo ""
echo "=== Kompilace Mytail ==="
cd ../Mytail
make clean
make
echo "Maytail zkompilován úspěšně"

echo ""
echo "=== Spuštění aplikací ==="
cd ..

echo "Spouštím generátor souborů..."
./FileGenerator/bin/data/generator -o test.txt -d 3

echo "Spouštím maytail..."
echo "Spustím maytail => ./Mytail/bin/maytail *.txt"

echo ""
echo "Zastavení mytail pomocí přízkazu => touch stop"