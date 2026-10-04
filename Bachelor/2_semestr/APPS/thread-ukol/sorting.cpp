#include <iostream>

void selectionSort(int arr[], int n) {
    for (int i = 0; i < n - 1; ++i) {
        // Najdenie indexu najmensieho prvku v zvysku pola
        int min_index = i;
        for (int j = i + 1; j < n; ++j) {
            if (arr[j] < arr[min_index]) {
                min_index = j;
            }
        }
        // Vymena najmensieho prvku s prvkom na pozicii i
        if (min_index != i) {
            std::swap(arr[i], arr[min_index]);
        }
    }
}

int main() {
    int arr[] = {64, 25, 12, 22, 11};
    int n = sizeof(arr) / sizeof(arr[0]);
    
    std::cout << "Nezotridene pole: ";
    for (int i = 0; i < n; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
    
    selectionSort(arr, n);
    
    std::cout << "Zotridene pole: ";
    for (int i = 0; i < n; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
