CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++11

TARGET = bin/maytail

# Zahrnout i maytail.cpp do zdrojových souborů
SOURCES = maytail.cpp FileUtil.cpp Monitor.cpp ParseArgs.cpp
OBJECTS = $(addprefix bin/, $(SOURCES:.cpp=.o))

all: $(TARGET)

bin:
	mkdir -p bin

$(TARGET): bin $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(TARGET)

bin/%.o: %.cpp | bin
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf bin

.PHONY: clean all