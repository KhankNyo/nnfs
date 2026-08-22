#ifndef NEURALNET_H
#define NEURALNET_H

#include <stdbool.h>

typedef struct neuralnet neuralnet;
typedef struct neuralnet_layer neuralnet_layer;
typedef struct neuralnet_config neuralnet_config;
typedef struct neuralnet_backprop_config neuralnet_backprop_config;
typedef struct neuralnet_feedforward_config neuralnet_feedforward_config;
typedef float (*neuralnet_activation_fn)(float Value);

/* memory allocation is not the focal point here, but since we're in C,
   it's kinda important to think about how you allocate memory */
typedef enum 
{
    NNALLOC_ALLOCATE,
    NNALLOC_FREE,
} neuralnet_allocator_mode;
typedef struct 
{
    neuralnet_allocator_mode Mode;
    union {
        struct {
            int SizeBytes;
        } Allocate;
        struct {
            void *Ptr;
        } Free;
    };
} neuralnet_allocator_param;
typedef void *(*neuralnet_allocator_callback)(void *AllocatorData, neuralnet_allocator_param *Param);


struct neuralnet_config
{
    int InputCount;

    int LayerCount;
    int *NodeCountPerLayer;

    /* again, not the focal point. Can provide NULL to AllocatorData and AllocatorCallback use the default allocators (malloc && free) */
    void *AllocatorData;
    neuralnet_allocator_callback AllocatorCallback;
};

struct neuralnet_backprop_config
{
    float LearningRate;
    int ExpectedOutputCount;
    const float *ExpectedOutputs;
};

struct neuralnet_feedforward_config
{
    neuralnet_activation_fn ActivationFn;
    const float *Inputs;
    int InputCount;
};


struct neuralnet
{
    int InputCount;
    float *Inputs;
    float *ScratchMatrix;

    int LayerCount;
    neuralnet_layer *Layers;

    void *AllocatorData;
    neuralnet_allocator_callback AllocatorCallback;
};

struct neuralnet_layer
{
    int InputCount;
    int OutputCount;
    /* [Input x Output matrix] */
    float *Weights;

    /* [array with length of OutputCount] */
    float *Biases;
    float *Outputs;
    float *Deltas;
};



neuralnet NeuralNet_Create(const neuralnet_config *Config);
neuralnet NeuralNet_CheapCopy(const neuralnet *NN);
void NeuralNet_Destroy(neuralnet *NN);

void NeuralNet_Randomize(neuralnet *NN);
void NeuralNet_FeedForward(neuralnet *NN, const neuralnet_feedforward_config *Config);
void NeuralNet_Backprop(neuralnet *NN, const neuralnet_backprop_config *Config);

void NeuralNet_Print(const neuralnet *NN);
float *NeuralNet_GetOutput(neuralnet *NN);



#endif /* NEURALNET_H */




#if defined(NEURALNET_IMPLEMENTATION) && !defined(NEURALNET_ALREADY_IMPLEMENTED)
#define NEURALNET_ALREADY_IMPLEMENTED


#include <stdlib.h> /* malloc, free */
#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <string.h>


#define NN__ALLOC(p_nn, size_bytes) \
    (p_nn)->AllocatorCallback(\
        (p_nn)->AllocatorData, \
        &(neuralnet_allocator_param) { \
            .Mode = NNALLOC_ALLOCATE, \
            .Allocate.SizeBytes = (size_bytes)\
        }\
    )
#define NN__FREE(p_nn, ptr) \
    (p_nn)->AllocatorCallback(\
        (p_nn)->AllocatorData,\
        &(neuralnet_allocator_param) {\
            .Mode = NNALLOC_FREE, \
            .Free.Ptr = (ptr), \
        }\
    )

#define NN__ALLOCATION_SCOPE_BEGIN(p_nn) 0
#define NN__ALLOCATION_SCOPE_END(p_nn) 0
#define NN__ALLOCATION_SCOPE(p_nn) for (\
        int nn__alloc_scope = (NN__ALLOCATION_SCOPE_BEGIN(p_nn), 1); \
        nn__alloc_scope; \
        NN__ALLOCATION_SCOPE_END(p_nn), (nn__alloc_scope = 0)\
    )
#define NN__MAX(a, b) ((a) > (b)? (a) : (b))

static void *NN__DefaultAllocatorCallback(void *Data, neuralnet_allocator_param *Param);
static void NN__LinearCombination(float *Y, const float *M, const float *X, const float *B, int Row, int Col);
static void NN__MatMulABT(float *Y, const float *A, const float *BT, int RowA, int ColA, int RowBT);
static void NN__MatTranpose(float *Result, const float *Mat, int Row, int Col);
static void NN__MatSubInPlace(float *Lhs, const float *Rhs, int Row, int Col);
static float NN__GetRandomValue(void);
static float NN__Sigmoid(float Value);
static float NN__SigmoidDerivativeY(float Y);


neuralnet NeuralNet_Create(const neuralnet_config *Config)
{
    assert(Config->LayerCount >= 1 && "must have at leaast 1 layer (output layer)");
    neuralnet NN = { 
        .InputCount = Config->InputCount,
        .LayerCount = Config->LayerCount,
    };
    if (Config->AllocatorCallback != NULL)
    {
        NN.AllocatorCallback = Config->AllocatorCallback;
        NN.AllocatorData = Config->AllocatorData;
    }
    else
    {
        NN.AllocatorCallback = NN__DefaultAllocatorCallback;
        NN.AllocatorData = NULL;
    }

    /* allocate needed mem */
    {
        NN.Inputs = NN__ALLOC(&NN, sizeof(NN.Inputs[0]) * NN.InputCount);
        NN.Layers = NN__ALLOC(&NN, sizeof(NN.Layers[0]) * NN.LayerCount);
        int InputCount = Config->InputCount;
        int LargestSide = InputCount;
        for (int i = 0; i < Config->LayerCount; i++)
        {
            int OutputCount = Config->NodeCountPerLayer[i];

            NN.Layers[i].Weights = NN__ALLOC(&NN, OutputCount*InputCount*sizeof(NN.Layers[0].Weights[0]));
            NN.Layers[i].Biases = NN__ALLOC(&NN, OutputCount*sizeof(NN.Layers[0].Biases[0]));
            NN.Layers[i].Outputs = NN__ALLOC(&NN, OutputCount*sizeof(NN.Layers[0].Outputs[0]));
            NN.Layers[i].Deltas = NN__ALLOC(&NN, OutputCount*sizeof(NN.Layers[0].Deltas[0]));
            NN.Layers[i].InputCount = InputCount;
            NN.Layers[i].OutputCount = OutputCount;

            LargestSide = NN__MAX(OutputCount, LargestSide);
            InputCount = OutputCount;
        }
        NN.ScratchMatrix = NN__ALLOC(&NN, LargestSide*LargestSide*sizeof(NN.ScratchMatrix[0]));
    }

    NeuralNet_Randomize(&NN);
    return NN;
}

neuralnet NeuralNet_CheapCopy(const neuralnet *NN)
{
    int *NodesPerLayer = NN__ALLOC(NN, NN->LayerCount * sizeof(NodesPerLayer[0]));
    for (int i = 0; i < NN->LayerCount; i++)
    {
        NodesPerLayer[i] = NN->Layers[i].OutputCount;
    }

    neuralnet_config Config = {
        .AllocatorCallback = NN->AllocatorCallback,
        .AllocatorData = NN->AllocatorData,
        .InputCount = NN->InputCount,
        .LayerCount = NN->LayerCount,
        .NodeCountPerLayer = NodesPerLayer,
    };
    neuralnet Result = NeuralNet_Create(&Config);

    NN__FREE(NN, NodesPerLayer);
    return Result;
}

void NeuralNet_Destroy(neuralnet *NN)
{
    for (int i = 0; i < NN->LayerCount; i++)
    {
        NN__FREE(NN, NN->Layers[i].Weights);
        NN__FREE(NN, NN->Layers[i].Deltas);
        NN__FREE(NN, NN->Layers[i].Biases);
        NN__FREE(NN, NN->Layers[i].Outputs);
    }
    NN__FREE(NN, NN->Layers);
    NN__FREE(NN, NN->Inputs);
    NN__FREE(NN, NN->ScratchMatrix);
}

void NeuralNet_FeedForward(neuralnet *NN, const neuralnet_feedforward_config *Config)
{
    /* NOTE: NeuralNet_Create() guarantees that 
     * consecutive layers have the previous layer's output node count == the current layer's input node count */
    assert(NN->InputCount == NN->Layers[0].InputCount);
    assert(NN->LayerCount >= 1);

    if (Config->InputCount)
    {
        assert(Config->InputCount == NN->InputCount);
        assert(Config->Inputs);
        memcpy(NN->Inputs, Config->Inputs, NN->InputCount * sizeof(NN->Inputs[0]));
    }
    neuralnet_activation_fn ActivationFn = Config->ActivationFn;
    if (ActivationFn == NULL)
    {
        ActivationFn = NN__Sigmoid;
    }


    const float *X = NN->Inputs;
    for (int i = 0; i < NN->LayerCount; i++)
    {
        neuralnet_layer *Layer = &NN->Layers[i];

        NN__LinearCombination(
            Layer->Outputs, 
            Layer->Weights, X, Layer->Biases,
            Layer->InputCount, Layer->OutputCount
        );

        /* NOTE: normalize outputs via activation fn ("squish" Y from -inf..+inf to 0..1) */
        for (int r = 0; r < Layer->OutputCount; r++)
        {
            Layer->Outputs[r] = ActivationFn(Layer->Outputs[r]);
        }

        X = Layer->Outputs;
    }
}

void NeuralNet_Backprop(neuralnet *NN, const neuralnet_backprop_config *Config)
{
    /* compute output layer deltas */
    {
        const neuralnet_layer *Last = NN->Layers + NN->LayerCount - 1;
        assert(Config->ExpectedOutputCount == Last->OutputCount);

        for (int i = 0; i < Last->OutputCount; i++)
        {
            float Error = Last->Outputs[i] - Config->ExpectedOutputs[i];
            Last->Deltas[i] = Error * NN__SigmoidDerivativeY(Last->Outputs[i]);
        }
    }

    /* compute hidden layer deltas */
    for (int i = NN->LayerCount - 2; i >= 0; i--)
    {
        neuralnet_layer *Curr = NN->Layers + i + 1;
        neuralnet_layer *Prev = NN->Layers + i;

        int Col = 1;
        NN__MatMulABT(
            Prev->Deltas, 
            Curr->Deltas, Curr->Weights, 
            Curr->OutputCount, Col, 
            Curr->InputCount
        );
        for (int k = 0; k < Prev->OutputCount; k++)
        {
            Prev->Deltas[k] *= Config->LearningRate * NN__SigmoidDerivativeY(Prev->Outputs[k]);
        }
    }

    /* update weights and biases */
    int InputCount = NN->InputCount;
    float *Inputs = NN->Inputs;
    for (int i = 0; i < NN->LayerCount; i++)
    {
        neuralnet_layer *Curr = NN->Layers + i;
        NN__MatMulABT(NN->ScratchMatrix, Curr->Deltas, Inputs, 1, Curr->OutputCount, InputCount);
        NN__MatSubInPlace(Curr->Weights, NN->ScratchMatrix, InputCount, Curr->OutputCount);
        NN__MatSubInPlace(Curr->Biases, Curr->Deltas, Curr->OutputCount, 1);

        Inputs = Curr->Outputs;
        InputCount = Curr->OutputCount;
    }
}


void NeuralNet_Print(const neuralnet *NN)
{
    printf("Inputs: [");
    for (int i = 0; i < NN->InputCount; i++)
    {
        printf("%g ", NN->Inputs[i]);
    }
    printf("]\n");

    printf("Layers: %d\n", NN->LayerCount);
    for (int i = 0; i < NN->LayerCount; i++)
    {
        const neuralnet_layer *Layer = NN->Layers + i;
        printf("    layer %d: in/out: %d/%d\n", i, Layer->InputCount, Layer->OutputCount);

        printf("        node vals:  [ ");
        for (int k = 0; k < Layer->OutputCount; k++)
            printf("%6.3f ", Layer->Outputs[k]);
        printf("]\n");

        printf("        node bias:  [ ");
        for (int k = 0; k < Layer->OutputCount; k++)
            printf("%6.3f ", Layer->Biases[k]);
        printf("]\n");

        printf("        node delta: [ ");
        for (int k = 0; k < Layer->OutputCount; k++)
            printf("%6.3f ", Layer->Deltas[k]);
        printf("]\n");

        printf("        weights:\n");
        for (int k = 0; k < Layer->OutputCount; k++)
        {
            printf("            [ ");
            for (int j = 0; j < Layer->InputCount; j++)
            {
                printf("%6.3f ", Layer->Weights[j + k*Layer->OutputCount]);
            }
            printf("]\n");
        }
    }

    printf("Neural network output: [ ");
    for (int i = 0; i < NN->Layers[NN->LayerCount - 1].OutputCount; i++)
        printf("%g ", NN->Layers[NN->LayerCount - 1].Outputs[i]);
    printf("]\n");
}

void NeuralNet_Randomize(neuralnet *NN)
{
    for (int n = 0; n < NN->LayerCount; n++)
    {
        neuralnet_layer *Layer = &NN->Layers[n];
        for (int k = 0; k < Layer->OutputCount; k++)
        {
            Layer->Biases[k] = NN__GetRandomValue();
            Layer->Outputs[k] = NN__GetRandomValue();
            Layer->Deltas[k] = NN__GetRandomValue();
            for (int i = 0; i < Layer->InputCount; i++)
            {
                Layer->Weights[k*Layer->OutputCount + i] = NN__GetRandomValue();
            }
        }
    }
}

float *NeuralNet_GetOutput(neuralnet *NN)
{
    return NN->Layers[NN->LayerCount - 1].Outputs;
}


static void *NN__DefaultAllocatorCallback(void *Data, neuralnet_allocator_param *Param)
{
    (void)Data;
    switch (Param->Mode)
    {
    case NNALLOC_ALLOCATE: return malloc(Param->Allocate.SizeBytes);
    case NNALLOC_FREE:     free(Param->Free.Ptr); break;
    }
    return NULL;
}

/* NOTE: Y = M . X + B
 * [Y_0]   [TM_00 TM_0r]   [X_0]   [B_0]
 * [Y_c] = [TM_c0 TM_cr] . [X_r] + [B_c]
 */
static void NN__LinearCombination(float *Y, const float *M, const float *X, const float *B, int Row, int Col)
{
    for (int c = 0; c < Col; c++)
    {
        float Tmp = 0;
        for (int r = 0; r < Row; r++)
        {
            Tmp += M[r + c*Row] * X[r];
        }
        Y[c] = Tmp + B[c];
    }
}

/* NOTE: Y = A . B^T */
static void NN__MatMulABT(float *Y, const float *A, const float *BT, int RowA, int ColA, int RowBT)
{
    for (int Rtb = 0; Rtb < RowBT; Rtb++)
    {
        for (int Ca = 0; Ca < ColA; Ca++)
        {
            float Tmp = 0;
            for (int Ra = 0; Ra < RowA; Ra++)
            {
                Tmp += A[Ra + Ca*RowA] * BT[Ra + Rtb*RowA];
            }
            Y[Ca*RowBT + Rtb] = Tmp;
        }
    }
}

static void NN__MatTranpose(float *Result, const float *Mat, int Row, int Col)
{
    for (int c = 0; c < Col; c++)
    {
        for (int r = 0; r < Row; r++)
        {
            Result[r*Col + c] = Mat[r + c*Row];
        }
    }
}

static void NN__MatSubInPlace(float *Lhs, const float *Rhs, int Row, int Col)
{
    for (int i = 0; i < Col; i++)
    {
        for (int k = 0; k < Row; k++)
        {
            int Index = k + i*Row;
            Lhs[Index] -= Rhs[Index];
        }
    }
}


static float NN__SigmoidDerivativeY(float Y)
{
    return Y*(1 - Y);
}

static float NN__GetRandomValue(void)
{
    return (float)rand() / RAND_MAX;
}

static float NN__Sigmoid(float Value)
{
    return 1.0 / (1.0 + expf(-Value));
}

#endif
