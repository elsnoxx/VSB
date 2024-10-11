#include <iostream>
#include <vector>
#include <algorithm>


using namespace std;

void PrintVector(const std::vector<int>& pole) {
    for (const auto& elem : pole) {
        std::cout << elem << " ";
    }
    std::cout << std::endl;
}


// Algoritmus hrubou silou (brute-force)
bool HasDuplicatesBruteForce(const std::vector<int>& pole) { // Pole jako reference
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



bool HasDuplicatesSorted(const std::vector<int>& pole) {

    std::vector<int> sortedPole = std::move(pole);
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
    std::cout << "Test 1 (brute-force, bez duplicit): " << (HasDuplicatesBruteForce(test1) ? "Unique" : "Duplicit") << std::endl;
    std::cout << "Test 2 (brute-force, s duplicitami): " << (HasDuplicatesBruteForce(test2) ? "Unique" : "Duplicit") << std::endl;

    // Testy pro algoritmus s předtříděním
    std::cout << "Test 3 (sorted, bez duplicit): " << (HasDuplicatesSorted(test1) ? "Unique" : "Duplicit") << std::endl;
    std::cout << "Test 4 (sorted, s duplicitami): " << (HasDuplicatesSorted(test2) ? "Unique" : "Duplicit") << std::endl;


    return 0;
}
