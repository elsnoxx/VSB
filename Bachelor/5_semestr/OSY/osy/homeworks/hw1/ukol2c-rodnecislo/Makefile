TARGET  = verbank
LIBNAME = validateNumber
LIBFILE = lib$(LIBNAME).so
LIBFILE1 = rodnecislo/lib$(LIBNAME).so

CPPFLAGS  += -g 
LDFLAGS += -L. 
LDLIBS += -l$(LIBNAME)

ifeq ($(shell uname -m),x86_64) 
 CPPFLAGS += -fPIC
endif

all: $(TARGET) $(LIBFILE1)

# clean binary files
clean:
	rm -rf $(TARGET) $(LIBFILE) $(LIBFILE1) *.o

# creating dynamic library verbank
$(LIBFILE): ${LIBNAME}.o
	@echo "\nCreating dynamic library...\n"
	g++ $(CPPFLAGS) $^ -shared $(LDFLAGS) -o $@
	@echo "\nDynamic library created: '$(LIBFILE)'.\n"

# creating dynamic library rodnecislo
$(LIBFILE1): rodnecislo/validateNumber.cpp | rodnecislo
	@echo "\nCreating dynamic library for rodnecislo...\n"
	g++ $(CPPFLAGS) -shared $< $(LDFLAGS) -o $@
	@echo "\nDynamic library created: '$(LIBFILE1)'.\n"

# creating application
$(TARGET): $(TARGET).cpp $(LIBFILE)
	@echo "\nCreating application...\n"
	g++ $(CFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@
	@echo "\nApplication '$(TARGET)' is ready.\n"
	@echo "Set path to library: 'export LD_LIBRARY_PATH=.' !\n"