CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++11

TARGET_GENERATOR = bin/data/generator

all: $(TARGET_GENERATOR)

bin/data:
	mkdir -p bin/data

$(TARGET_GENERATOR): bin/data generatro.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET_GENERATOR) generatro.cpp

clean:
	rm -f $(TARGET_GENERATOR)
	rm -rf bin

.PHONY: clean all