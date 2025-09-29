CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++11 -fPIC
LDFLAGS = -shared

TARGET2 = bin/verbank
VALSRC  = validateNumber.cpp
VALSRC1 = rodnecislo/validateNumber.cpp

DYNLIB  = lib/libvalidateNumber.so
DYNLIB1 = rodnecislo/libvalidateNumber.so

all: $(DYNLIB) $(DYNLIB1) $(TARGET2)

# Vytvoření složek bin a lib, pokud neexistují
bin:
	mkdir -p bin
	mkdir -p lib

# Kompilace dynamické knihovny accountValidator
$(DYNLIB): $(VALSRC) | bin
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(DYNLIB) $(VALSRC)

# Kompilace dynamické knihovny rodnecislo
$(DYNLIB1): $(VALSRC1) | bin
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(DYNLIB1) $(VALSRC1)

# Překlad programu verbank
$(TARGET2): verbank.cpp $(DYNLIB) | bin
	$(CXX) $(CXXFLAGS) -o $(TARGET2) verbank.cpp -Llib -lvalidateNumber

clean:
	rm -f $(TARGET2) $(DYNLIB) $(DYNLIB1)
	rmdir bin 2>/dev/null || true
	rmdir lib 2>/dev/null || true

.PHONY: all clean bin