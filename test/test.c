
#define NEURALNET_IMPLEMENTATION
#include "neuralnet.h"
#include <stdio.h>

static void MatPrint(const float *M, int Row, int Col)
{
    for (int c = 0; c < Col; c++)
    {
        printf("[");
        for (int r = 0; r < Row; r++)
        {
            printf("%6.3f ", M[r + c*Row]);
        }
        printf("]\n");
    }
}

int main(void)
{
    printf("\ntest1:\n");
    {
        float A[] = {
            1, 2, 3, 
            4, 5, 6,
        };
        float B[] = {
            6, 5, 4, 
            3, 2, 1,
            7, 8, 9
        };
        float BT[3*3];
        float Result[2*3] = { 0 };
        NN__MatTranpose(BT, B, 3, 3);
        NN__MatDot(Result, A, BT, 3, 2, 3);
        MatPrint(Result, 3, 2);
    }

    printf("\ntest2:\n");
    {
        float Result[3] = { 0 };
        float M[] = {
            1, 2,
            4, 5,
        };
        float X[] = {
            1, 
            4
        };
        float B[] = {
            0, 
            0
        };
        NN__LinearCombination(Result, M, X, B, 2, 2);
        MatPrint(Result, 2, 1);
    }

    return 0;
}
