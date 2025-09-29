CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++11 -fPIC
LDFLAGS = -shared

TARGET2 = bin/verbank
VALSRC  = validateNumber.cpp

DYNLIB  = lib/libvalidateAcc.so

all: $(DYNLIB) $(TARGET2)

# Vytvoření složek bin a lib, pokud neexistují
bin:
	mkdir -p bin
	mkdir -p lib

# Kompilace dynamické knihovny
$(DYNLIB): $(VALSRC) | bin
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(DYNLIB) $(VALSRC)

# Překlad programu verbank
$(TARGET2): verbank.cpp $(DYNLIB) | bin
	$(CXX) $(CXXFLAGS) -o $(TARGET2) verbank.cpp -Llib -lvalidateAcc

clean:
	rm -f $(TARGET2) $(DYNLIB)
	rmdir bin 2>/dev/null || true
	rmdir lib 2>/dev/null || true

.PHONY: all clean bin