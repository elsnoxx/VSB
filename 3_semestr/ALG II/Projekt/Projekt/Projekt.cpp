#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>


using namespace std;


int main()
{
    ifstream file("C:\\Users\\admin\\Documents\\GitHub\\VSB\\3_semestr\\ALG II\\Projekt\\Data\\test0.txt");

    string myText;

    while (getline(file, myText)) {
        // Output the text from the file
        cout << myText << "\n";
    }

    int age;

    cout << "Enter your age: ";

    // get age from user
    scanf_s("%d", &age);

    // print age
    cout << "Age = " << age;

    return 0;
}