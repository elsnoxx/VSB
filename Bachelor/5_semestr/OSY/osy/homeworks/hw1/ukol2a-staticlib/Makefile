TARGET = gennum
LIBNAME = generator
LIBFILE = lib$(LIBNAME).a

CPPFLAGS += -g
LDFLAGS += -L. 
LDLIBS += -l$(LIBNAME)

all: $(TARGET)

# clean binary files
clean:
	rm -rf $(TARGET) $(LIBFILE) *.o

# creating static library 
$(LIBFILE): ${LIBNAME}.o 
	@echo "\nCreating static library...\n"
	#rm -f $@
	ar r $@ $^
	@echo "\nStatic library created: '$(LIBFILE)'.\n"

# creating application
$(TARGET): $(TARGET).cpp $(LIBFILE)
	@echo "\nCreating application...\n"
	g++ $(CFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@
	@echo "\nApplication '$(TARGET)' is ready.\n"