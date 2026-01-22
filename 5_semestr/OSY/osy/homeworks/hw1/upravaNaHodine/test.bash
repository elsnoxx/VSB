#!/bin/bash

echo "export LD_LIBRARY_PATH=$(pwd)/"
export LD_LIBRARY_PATH=$(pwd)/
./gennum 2510040000 10000 | ./verbank -v | wc 
./gennum 2560040000 10000 | ./verbank -v | wc 

echo "export LD_LIBRARY_PATH=$(pwd)/rodnecislo"
export LD_LIBRARY_PATH=$(pwd)/rodnecislo
./gennum 2510040000 10000 | ./verbank -v | wc 
./gennum 2560040000 10000 | ./verbank -v | wc 

echo "export LD_LIBRARY_PATH=$(pwd)/treti_knihovna"
export LD_LIBRARY_PATH=$(pwd)/treti_knihovna
./gennum 2510040000 10000 | ./verbank -v | wc
./gennum 2560040000 10000 | ./verbank -v | wc
