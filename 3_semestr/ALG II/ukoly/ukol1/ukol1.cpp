#include <iostream>
#include <vector>
#include <algorithm>


using namespace std;

void PrintVector(std::vector<int>& pole) {
    int size = pole.size();
    for (int i = 0; i <= size - 1; i++) {
        std::cout << pole[i] << " ";
    }
    std::cout << " " << pole.size() << std::endl;
}

// Algoritmus hrubou silou (brute-force)
bool UniqueElementsBruteForce(const std::vector<int>& pole) { // Pole jako reference
    int size = pole.size();
    for (int i = 0; i < size - 1; i++) {
        for (int j = i + 1; j < size; j++) {
            if (pole[i] == pole[j]) {
                return false; // Duplicitní prvek nalezen
            }
        }
    }
    return true; // Žádné duplicity
}



bool UniqueElements(const std::vector<int>& pole) {

    std::vector<int> sortedPole = pole;
    sort(sortedPole.begin(), sortedPole.end());

    int size = sortedPole.size();
    for (int i = 0; i < size - 1; i++) {
        if (sortedPole[i] == sortedPole[i + 1]) {
            return false;
        }
    }
    return true;
}



int main() {
    // Bez duplicit
    std::vector<int> test1 = { 10, 4, 5, 2, 1, 7, 9, 6, 8, 3 };
    // S duplicitou
    std::vector<int> test2 = { 5, 2, 8, 7, 6, 4, 10, 9, 10, 3 };

    // Testy pro hrubou silu
    std::cout << "Test 1 (brute-force, bez duplicit): " << (UniqueElementsBruteForce(test1) ? "Unique" : "Duplicit") << std::endl;
    std::cout << "Test 2 (brute-force, s duplicitami): " << (UniqueElementsBruteForce(test2) ? "Unique" : "Duplicit") << std::endl;

    // Testy pro algoritmus s předtříděním
    std::cout << "Test 3 (sorted, bez duplicit): " << (UniqueElements(test1) ? "Unique" : "Duplicit") << std::endl;
    std::cout << "Test 4 (sorted, s duplicitami): " << (UniqueElements(test2) ? "Unique" : "Duplicit") << std::endl;


    return 0;
}
