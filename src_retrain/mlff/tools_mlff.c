/**
 * @file    tools.c
 * @brief   This file contains the tool functions.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>
#include <math.h>

#include "mlff_types.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

// /**
//  * @brief   Solves a tridiagonal system using Gauss Elimination.
//  */
// void tridiag_gen(double *A, double *B, double *C, double *D, int len) {
//     int i;
//     double b, *F;
//     F = (double *)malloc(sizeof(double)*len);
//     if (F == NULL) {
//         printf("Memory allocation failed!\n");
//         exit(EXIT_FAILURE);
//     }
    
//     // Gauss elimination; forward substitution
//     b = B[0];
//     if (fabs(b) < 1e-14) {
//         printf("Divide by zero in tridiag_gen\n"); 
//         exit(EXIT_FAILURE);
//     }
//     D[0] = D[0]/b;
//     for (i = 1; i<len; i++) {
//         F[i] = C[i-1] / b;
//         b= B[i] - A[i] * F[i];
//         if (fabs(b) < 1e-14) {
//             printf("Divide by zero in tridiag_gen\n"); 
//             exit(EXIT_FAILURE);
//         }
//         D[i] = (D[i] - D[i-1] * A[i])/b;
//     }
//     // backsubstitution 
//     for (i = len-2; i >= 0; i--)
//         D[i] -= (D[i+1]*F[i+1]);
        
//     free(F);
// }


// /**
//  * @brief   Calculates derivatives of a tabulated function required for spline interpolation.
//  */
// void getYD_gen(double *X, double *Y, double *YD, int len) {
//     int i;
//     double h0,h1,r0,r1,*A,*B,*C;
          
//     A = (double *)malloc(sizeof(double)*len);
//     B = (double *)malloc(sizeof(double)*len);
//     C = (double *)malloc(sizeof(double)*len);
//     if (A == NULL || B == NULL || C == NULL) {
//         printf("Memory allocation failed!\n");
//         exit(EXIT_FAILURE);
//     }

//     h0 =  X[1]-X[0]; h1 = X[2]-X[1];
//     r0 = (Y[1]-Y[0])/h0; r1=(Y[2]-Y[1])/h1;
//     B[0] = h1*(h0+h1);
//     C[0] = (h0+h1)*(h0+h1);
//     YD[0] = r0*(3*h0*h1 + 2*h1*h1) + r1*h0*h0;
               
//     for(i=1;i<len-1;i++) {
//         h0 = X[i]-X[i-1]; h1=X[i+1]-X[i];
//         r0 = (Y[i]-Y[i-1])/h0;  r1=(Y[i+1]-Y[i])/h1;
//         A[i] = h1;
//         B[i] = 2*(h0+h1);
//         C[i] = h0;
//         YD[i] = 3*(r0*h1 + r1*h0);
//     }
           
//     A[i] = (h0+h1)*(h0+h1);
//     B[i] = h0*(h0+h1);
//     YD[i] = r0*h1*h1 + r1*(3*h0*h1 + 2*h0*h0);
     
//     tridiag_gen(A,B,C,YD,len);
    
//     free(A); free(B); free(C);                                     
// }







// /**
//  * @brief   Cubic spline evaluation from precalculated data.
//  */
// void SplineInterp(double *X1,double *Y1,int len1,double *X2,double *Y2,int len2,double *YD) {
//     if (len2 <= 0) return;
//     int i,j;
//     double A0,A1,A2,A3,x,dx,dy,p1,p2,p3;    
//     if(X2[0]<X1[0] || X2[len2-1]>X1[len1-1]) {
//         printf("First input X in table=%lf, last input X in table=%lf, "
//                "interpolate at x[first]=%lf, x[last]=%lf\n",X1[0],X1[len1-1],X2[0],X2[len2-1]);
//         printf("Out of range in spline interpolation!\n");
//         exit(EXIT_FAILURE);
//     }
//     // p1 is left endpoint of the interval
//     // p2 is resampling position
//     // p3 is right endpoint of interval
//     // j is input index of current interval  
//     A0 = A1 = A2 = A3 = 0.0;
//     p1 = p3 = X2[0]-1;  // force coefficient initialization  
//     for (i = j = 0; i < len2; i++) {
//         // check if in new interval
//         p2 = X2[i];
//         if (p2 > p3) {
//             //find interval which contains p2 
//             for (; j<len1 && p2>X1[j]; j++);
//             if (p2 < X1[j]) j--;
//             p1 = X1[j]; 
//             p3 = X1[j+1]; 
//             // coefficients
//             dx = 1.0 / (X1[j+1] - X1[j]);
//             dy = (Y1[j+1] - Y1[j]) * dx;
//             A0 = Y1[j];
//             A1 = YD[j];
//             A2 = dx * (3.0 * dy - 2.0 * YD[j] - YD[j+1]);
//             A3 = dx * dx * (-2.0*dy + YD[j] + YD[j+1]);  
//         }
//         // use Horner's rule to calculate cubic polynomial
//         x = p2-p1;
//         Y2[i] = ((A3*x + A2) * x + A1) * x + A0;     
//     } 
// }


// /**
//  * @brief Initialize the dynamic array, allocate initial
//  *        memory and set size.
//  *
//  */
 
// #define INIT_CAPACITY 4
// void init_dyarray(dyArray *a)
// {
//     assert(a != NULL);
//     size_t initsize = INIT_CAPACITY;
//     a->array = malloc(initsize * sizeof(*a->array));
//     assert(a->array != NULL);
//     a->len = 0;
//     a->capacity = initsize;
// }

// /**
//  * @brief Realloc the dynamic array to the given new capacity.
//  *
//  *        Note that if the array is extended, all the previous data
//  *        are still preserved. If the array is shrinked, all the
//  *        previous data up to the new capacity is preserved.
//  */
// void realloc_dyarray(dyArray *a, size_t new_capacity)
// {
//     assert(a != NULL);
//     value_type *new_arr = realloc(a->array, new_capacity * sizeof(*a->array));
//     assert(new_arr != NULL);
//     a->array = new_arr;
//     a->capacity = new_capacity;
// }

// /**
//  * @brief Double the capacity of the dynamic array.
//  *
//  *        Note that if the array is extended, all the previous data
//  *        are still preserved. If the array is shrinked, all the
//  *        previous data up to the new capacity is preserved.
//  */
// void dyarray_double_capacity(dyArray *a) {
//     assert(a != NULL);
//     size_t new_capacity = a->capacity ? a->capacity << 1 : INIT_CAPACITY;
//     new_capacity = max(new_capacity, INIT_CAPACITY);
//     realloc_dyarray(a, new_capacity);
// }

// /**
//  * @brief Half the capacity of the dynamic array.
//  *
//  *        Note that if the array is extended, all the previous data
//  *        are still preserved. If the array is shrinked, all the
//  *        previous data up to the new capacity is preserved.
//  */
// void dyarray_half_capacity(dyArray *a) {
//     assert(a != NULL);
//     size_t new_capacity = a->capacity >> 1;
//     new_capacity = max(new_capacity, INIT_CAPACITY);
//     realloc_dyarray(a, new_capacity);
// }

// /**
//  * @brief Append an element to the dynamic array.
//  *
//  */
// void append_dyarray(dyArray *a, value_type element)
// {
//     assert(a != NULL);

//     // double the size of memory allocated if it's len up
//     if (a->len == a->capacity) {
//         dyarray_double_capacity(a);
//     }

//     // append the element to array
//     a->array[a->len] = element;
//     a->len++;
// }

// /**
//  * @brief Pop the last element from the dynamic array.
//  *
//  */
// value_type pop_dyarray(dyArray *a)
// {
//     assert(a != NULL);
//     if (a->len < 1) {
//         printf("Error: pop_dyarray target is empty!\n");
//         exit(1);
//     }

//     a->len--; // reduce len by 1

//     if (4 * a->len < a->capacity) {
//         dyarray_half_capacity(a);
//     }

//     return a->array[a->len];
// }

// /**
//  * @brief Clear the dynamic array.
//  *
//  *        This function does not destroy the array, it simply
//  *        resets the lenght of the dynamic array to 0, and resets
//  *        the capacity.
//  */
// void clear_dyarray(dyArray *a) {
//     assert(a != NULL);
//     size_t initsize = INIT_CAPACITY;
//     realloc_dyarray(a, initsize);
//     a->len = 0;
// }

// /**
//  * @brief Delete the dynamic array.
//  *
//  */
// void delete_dyarray(dyArray *a)
// {
//     assert(a != NULL);
//     free(a->array);
//     a->array = NULL;
//     a->len = a->capacity = 0;
// }


// /**
//  * @brief Print the dynamic array.
//  *
//  */
// void print_dyarray(const dyArray *a) {
//     printf("([");
//     for (int i = 0; i < a->len; i++) {
//         if (i > 0) printf(" ");
//         printf("%d", a->array[i]);
//     }
//     printf("], len = %ld, capacity = %ld)\n",a->len,a->capacity);
// }

// // if array is too long, only show the first 5 and last 5
// void show_dyarray(const dyArray *a) {
//     if (a->len <= 10) {
//         print_dyarray(a);
//         return;
//     }

//     printf("([");
//     for (int i = 0; i < 5; i++) {
//         if (i > 0) printf(" ");
//         printf("%d", a->array[i]);
//     }
//     printf(" ...");
//     for (int i = a->len-5; i < a->len; i++) {
//         if (i > 0) printf(" ");
//         printf("%d", a->array[i]);
//     }
//     printf("], len = %ld, capacity = %ld)\n",a->len,a->capacity);
// }

/*
dotProduct function returns the dot product of two arrays.

[Input]
1. vect_A: pointer to first array
2. vect_B: pointer to second array
3. size_vector: size of the vector
[Output]
1. product: dot product
*/

double dotProduct(double* vect_A, double* vect_B, int size_vector)
{
    double product = 0;
 
    // Loop for calculate cot product
    for (int i = 0; i < size_vector; i++)
        product = product + vect_A[i] * vect_B[i];

    return product;
}

/*
get_mean function returns the mean of values in an array.

[Input]
1. a: pointer to  array
2. n: size of array
[Output]
1. mean: mean
*/


double get_mean(double a[], int n)
{
    // Compute mean (average of elements)
    double sum = 0;
    for (int i = 0; i < n; i++)
        sum += a[i];
    double mean = (double)sum /
                  (double)n;
 
    return mean;
}

/*
get_variance function returns the variance of values in an array.

[Input]
1. a: pointer to  array
2. n: size of array
[Output]
1. sqDiff / (n-1): variance
*/

double get_variance(double a[], int n)
{
    // Compute mean (average of elements)
    double mean = get_mean(a,n);
    // Compute sum squared
    // differences with mean.
    double sqDiff = 0.0;
    for (int i = 0; i < n; i++)
        sqDiff += (a[i] - mean) *
                  (a[i] - mean);
    return sqDiff / (n-1);
}

double largest(double* arr, int n)
{
    int i;
    
    // Initialize maximum element
    double max = arr[0];
 
    // Traverse array elements from second and
    // compare every element with current max 
    for (i = 1; i < n; i++)
        if (arr[i] > max)
            max = arr[i];
 
    return max;
}

double smallest(double* arr, int n)
{
    int i;
    
    // Initialize maximum element
    double min = arr[0];
 
    // Traverse array elements from second and
    // compare every element with current max 
    for (i = 1; i < n; i++)
        if (arr[i] < min)
            min = arr[i];
 
    return min;
}



/*
lin_search function searches for the first occurence of an element in the array 

[Input]
1. arr: pointer to array
2. n: length of array
3. x: element to search

[Output]
1. i: ID of the element in the array
*/

double lin_search_double(double *arr, int n, double x)
{
    int i;
    for (i = 0; i < n; i++)
        if (arr[i] == x)
            return i;
    return -1;
}

double lin_search_INT(int *arr, int n, int x)
{
    int i;
    for (i = 0; i < n; i++)
        if (arr[i] == x)
            return i;
    return -1;
}