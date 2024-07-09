// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Parallel Systems
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava, 2020/11
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology usage with unified memory.
//
// Multiplication of elements in float array.
//
// ***********************************************************************

#include <stdio.h>
#include <math.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>
#include <string>
#include <typeinfo>

// Function prototype from .cu file
void cu_run_sum( float *t_array, float *t_array2, float *result, int t_length, float t_mult );
void cu_run_prevod(char *buffer, long t_length);
void cu_run_matice(double *MaticeA,double *MaticeB,double *MaticeC, int t_length);


void printArray(float *l_array, int N){
    for ( int i = 0; i < N; i++ )
        printf( "%8.2f", l_array[ i ] );
    printf( "\n" );

}

void checkArray(float *l_array, float *l_array2, int N)
{
   int konec = 0;
   for (int i = 0; i < N; i++)
   {
	if (l_array[i] != l_array2[i] * 2)
	{
	   konec = 1;
	   break;
	}
   }

   if (konec == 1)
   {
	printf("Not same\n");
   }else{
        printf("same\n");
   }
}

void task1(int N)
{
    // Array initialization 
    float *l_array;
    if ( cudaMallocManaged( &l_array, N * sizeof( *l_array ) ) != cudaSuccess )
    {
        printf( "Unable to allocate Unified memory!\n" );
    }

    float *l_array2;
    if (cudaMallocManaged( &l_array2, N * sizeof(*l_array2)) != cudaSuccess)
    {
        printf( "Unable to allocate Unified memory!\n" );
    }
    for ( int i = 0; i < N; i++ )
    {
        l_array[ i ] = ( float ) i;
    }
    for ( int i = 0; i < N; i++ )
    {
        l_array2[ i ] = ( float)  i;
    }

    float *result = new float[N];
    if (cudaMallocManaged( &result, N * sizeof(*result)) != cudaSuccess)
    {
        printf( "Unable to allocate Unified memory!\n" );
    }

    //printArray(l_array, N);
    //printArray(l_array2, N);

    for (int i = 0; i< N;i++)
    {
        result[i] = 0.0;
    }
    
    // Function calling from .cu file
    cu_run_sum( l_array, l_array2, result,  N, M_PI );
    
    checkArray(result, l_array2, N);
    

    cudaFree( l_array );
    cudaFree( l_array2);
    cudaFree( result );

}

void task2()
{
    const char* source_file_name = "main2_unm.cpp";

    // Otevreni souboru
    FILE* source_file = fopen(source_file_name, "r");
    if (!source_file) {
        std::cerr << "NUnable to open souce file!!" << std::endl;
    }

    // Ziskani delky souboru
    fseek(source_file, 0, SEEK_END);
    long file_size = ftell(source_file);
    fseek(source_file, 0, SEEK_SET);

    // Alokace pameti pro nacteni obsahu souboru
    char* buffer;
    if (cudaMallocManaged(&buffer, (file_size + 1) * sizeof(file_size)) != cudaSuccess)
    {
        printf("Unable to allocate Unified memory!\n");
    }
    
    // Precteni obsahu souboru do bufferu
    size_t read_bytes = fread(buffer, 1, file_size, source_file);
    buffer[read_bytes] = '\0';

    // Uzavreni zdrojoveho souboru
    fclose(source_file);
    // volani cudy
    cu_run_prevod(buffer, file_size + 1);
    
    // Ulozeni obsahu do vystupniho souboru
    std::string source_code(buffer);
    FILE* output_file = fopen("source_code.up", "w");
    if(!output_file)
    {
	std::cerr << "Nepodarilo se otevrit soubor" << std::endl;
    }
    fwrite(buffer, 1, read_bytes, output_file);
    fclose(output_file);
    // Uvolneni pameti
   
    
    cudaFree( buffer );
}
void initMatrix(double *matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i * cols + j] = rand() % 10;
        }
    }
}
void initMatrixE(double *matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i * cols + j] = 0;
        }
    }
}

void printMatrix(double *matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

void transposeMatrix(double *src, double *dst, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}


void task3(int M){
    printf("Task3 %d\n",M);
    
    double *MaticeA;
    double *MaticeB;
    double *MaticeC;
    double *MaticeB_T;
    if (cudaMallocManaged(&MaticeA, (M * M) * sizeof(double)) != cudaSuccess)
    {
        printf("Unable to allocate Unified memory!\n");
    }
    if (cudaMallocManaged(&MaticeB, (M * M) * sizeof(double)) != cudaSuccess)
    {
        printf("Unable to allocate Unified memory!\n");
    }
    if (cudaMallocManaged(&MaticeC, (M * M) * sizeof(double)) != cudaSuccess)
    {
        printf("Unable to allocate Unified memory!\n");
    }
    if (cudaMallocManaged(&MaticeB_T, (M * M) * sizeof(double)) != cudaSuccess) {
        printf("Unable to allocate Unified memory!\n");
    }
    
    printf("ok\n"); 
    initMatrix(MaticeA, M, M);
    initMatrix(MaticeB, M, M);

    transposeMatrix(MaticeB, MaticeB_T, M, M);

    
    initMatrixE(MaticeC, M, M);

    cu_run_matice(MaticeA, MaticeB_T, MaticeC, M);

    printMatrix(MaticeA, M,M);
    printf("\n");
    printMatrix(MaticeB_T, M,M);
    
    printf("\n\n");
    printMatrix(MaticeC, M,M);
    
    cudaFree(MaticeA);
    cudaFree(MaticeB);
    cudaFree(MaticeC);
}
// hodnota pro delku pole
#define N 100000000


// hodnata pro velikost matice
#define M 1000

int main()
{
    // Task 1 - 
    // Dle prikladu cuda2_unm si naimplementujte funkci a kernel, ktery provede secteni dvou vektoru prvku float a vysledek se vrati ve tretim vektoru.
    // Delka vektoru minimalne 1000000 prvku. Zvolte si prvky vektoru tak, aby jste nasledne dokazali snadno v programu zkontrolovat, ze vysledek je spravne
    // Ne ale primitivni zadani typu: vsechny prvky pole jsou 0 nebo 1, nebo vsechny stejne.
    std::cout << std::endl<< "Task 1 secteni dvou poli, implementovana i kontrola zdali vysledek je spravny."<<std::endl;
    task1(N);

    //Task 2 -
    // Napiste si funkci a kernel, ktery prevede retezec na velka pismena. Nactete si svuj zdrojovy kod *.cpp, prevedte ho na velka pismena a ulozte jako *.up.
    // Pro prevod na velka pismena si pro kernel pripravte vhodne pole znaku (konverzni tabulku), aby kernel neobsahoval pro samotny prevod zadny if.
    std::cout << std::endl<< "Task 2 prevod na velke pismena, generovani souboru source_code.up"<<std::endl;
    task2();
    std::cout << std::endl<< "Soubor \"source_code.up\" vygenerovan."<<std::endl;
    // Task 3 -
    // Napiste si funkci a kernel pro nasobeni dvou matic a vysledek se ulozi do treti matice. Rozmer matic minimalne 1000x1000 a typ prvku double. Matice bude mit
    // pevne danou velikost a bude tak tvorit jeden souvisly blok dat. Druhou matici si pred nasobenim transponujte, aby nasobeni neprobihalo standardne radek x 
    // sloupec, ale radek x radek. Kernel bude obsahovat jedinou smycku for pro vypocet jednoho prvku matice (vsechny kernely tak budou vykonavat stejny kod).
    // Prvky v obou maticich si pripravte tak, aby se nasledne dalo snadno (automaticky) zkontrolovat, ze vysledek je spravne. 
    // (Ne napr. vsechny prvky matice 0 nebo 1 a ve vysledku vsechny prvky stejne hodnoty).
    std::cout << std::endl<< "Task 3 nasobeni matic"<<std::endl;
    task3(M);
    return 0;
}