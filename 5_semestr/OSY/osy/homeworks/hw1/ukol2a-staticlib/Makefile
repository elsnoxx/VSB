CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++11

TARGET1 = bin/gennum
TARGET2 = bin/verbank

SOURCE1 = gennum.cpp
SOURCE2 = verbank.cpp
GENSRC  = generator.cpp

GENLIB  = lib/libgenerator.a
GENOBJ  = lib/generator.o

all: $(GENLIB) $(TARGET1) $(TARGET2)

# Vytvoření složek bin a lib, pokud neexistují
bin:
	mkdir -p bin
	mkdir -p lib

# kompilace statické knihovny
$(GENLIB): $(GENOBJ)
	ar rcs $(GENLIB) $(GENOBJ)

$(GENOBJ): $(GENSRC) | bin
	$(CXX) $(CXXFLAGS) -c $(GENSRC) -o $(GENOBJ)

# gennum linkuje knihovnu
$(TARGET1): $(SOURCE1) $(GENLIB) | bin
	$(CXX) $(CXXFLAGS) -o $(TARGET1) $(SOURCE1) -Llib -lgenerator

# druhý program
$(TARGET2): $(SOURCE2) | bin
	$(CXX) $(CXXFLAGS) -o $(TARGET2) $(SOURCE2)

clean:
	rm -f $(TARGET1) $(TARGET2) $(GENOBJ) $(GENLIB)
	rmdir bin 2>/dev/null || true
	rmdir lib 2>/dev/null || true

.PHONY: all clean bin