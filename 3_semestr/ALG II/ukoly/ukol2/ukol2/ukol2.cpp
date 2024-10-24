

#include <iostream>

class State {
    bool prevoznik;
    bool koza;
    bool zeli;
    bool vlk;

    State(bool boatman, bool wolf, bool goat, bool cabbage) : prevoznik(boatman), vlk(wolf), koza(goat), zeli(cabbage) {}

    bool isValid() {
        if (vlk && koza) {
            return false;
        }

        else if (koza && zeli) {
            return false;
        }

        else
        {
            return true;
        }
    }

    

};

int main()
{
    std::cout << "Hello World!\n";
}
