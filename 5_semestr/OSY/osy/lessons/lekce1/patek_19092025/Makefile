CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++11
TARGET = main
SOURCE = main.cpp
TARGET1 = scitacka
SOURCE1 = scitacka.cpp


all: $(TARGET) $(TARGET1)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCE)

$(TARGET1): $(SOURCE1)
	$(CXX) $(CXXFLAGS) -o $(TARGET1) $(SOURCE1)

clean:
	rm -f $(TARGET) $(TARGET1)

.PHONY: all clean