CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++11

TARGET1 = bin/gennum
TARGET2 = bin/verbank

SOURCE1 = gennum.cpp
SOURCE2 = verbank.cpp

all: $(TARGET1) $(TARGET2)

# Vytvoření složky bin, pokud neexistuje
bin:
	mkdir -p bin

# Překlad programu gennum
$(TARGET1): $(SOURCE1) bin
	$(CXX) $(CXXFLAGS) -o $(TARGET1) $(SOURCE1)

# Překlad programu verbank
$(TARGET2): $(SOURCE2) bin
	$(CXX) $(CXXFLAGS) -o $(TARGET2) $(SOURCE2)

clean:
	rm -f $(TARGET1) $(TARGET2)
	rmdir bin 2>/dev/null || true

.PHONY: all clean bin