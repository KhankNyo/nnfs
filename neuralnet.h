#ifndef NEURALNET_H
#define NEURALNET_H

#include <stdbool.h>

typedef struct neuralnet neuralnet;
typedef struct neuralnet_layer neuralnet_layer;
typedef struct neuralnet_config neuralnet_config;
typedef struct neuralnet_backprop_config neuralnet_backprop_config;
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
    float *ExpectedOutputs;
};


struct neuralnet
{
    int InputCount;
    float *Inputs;

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
    float *NodeBiases;
    float *NodeValues;
    float *DeltaWeight;
    float *DeltaBias;
};



neuralnet NeuralNet_Create(const neuralnet_config *Config);
void NeuralNet_Destroy(neuralnet *NN);
void NeuralNet_FeedForward(neuralnet *NN, neuralnet_activation_fn Fn);
/* NOTE: to use backprop, activation function used for feed forward must be differentiable (no step fn, ex: RELU) */
void NeuralNet_BackPropagate(neuralnet *NN, neuralnet_backprop_config *Config);



#endif /* NEURALNET_H */




#if defined(NEURALNET_IMPLEMENTATION) && !defined(NEURALNET_ALREADY_IMPLEMENTED)
#define NEURALNET_ALREADY_IMPLEMENTED


#include <stdlib.h> /* malloc, free */
#include <stdint.h>
#include <assert.h>
#include <time.h>
#include <math.h>


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

static void *NN__DefaultAllocatorCallback(void *Data, neuralnet_allocator_param *Param);
static void NN__LinearCombination(float *Y, const float *M, const float *X, const float *B, int Row, int Col);
static void NN__MatDot(float *Y, const float *A, const float *BT, int RowA, int ColA, int RowBT);
static void NN__MatTranpose(float *Result, const float *Mat, int Row, int Col);
static void NN__Randomize(neuralnet *NN);
static float NN__GetRandomValue(void);
static float NN__StepFn(float Value);
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
        for (int i = 0; i < Config->LayerCount; i++)
        {
            int OutputCount = Config->NodeCountPerLayer[i];

            NN.Layers[i].Weights = NN__ALLOC(&NN, OutputCount*InputCount*sizeof(NN.Layers[0].Weights[0]));
            NN.Layers[i].NodeBiases = NN__ALLOC(&NN, OutputCount*sizeof(NN.Layers[0].NodeBiases[0]));
            NN.Layers[i].NodeValues = NN__ALLOC(&NN, OutputCount*sizeof(NN.Layers[0].NodeValues[0]));
            NN.Layers[i].DeltaWeight = NN__ALLOC(&NN, OutputCount*sizeof(NN.Layers[0].DeltaWeight[0]));
            NN.Layers[i].DeltaBias = NN__ALLOC(&NN, OutputCount*sizeof(NN.Layers[0].DeltaBias[0]));
            NN.Layers[i].InputCount = InputCount;
            NN.Layers[i].OutputCount = OutputCount;

            InputCount = OutputCount;
        }
    }

    NN__Randomize(&NN);
    return NN;
}

void NeuralNet_Destroy(neuralnet *NN)
{
    for (int i = 0; i < NN->LayerCount; i++)
    {
        NN__FREE(NN, NN->Layers[i].Weights);
        NN__FREE(NN, NN->Layers[i].NodeBiases);
        NN__FREE(NN, NN->Layers[i].NodeValues);
        NN__FREE(NN, NN->Layers[i].DeltaWeight);
        NN__FREE(NN, NN->Layers[i].DeltaBias);
    }
    NN__FREE(NN, NN->Layers);
    NN__FREE(NN, NN->Inputs);
}

void NeuralNet_FeedForward(neuralnet *NN, neuralnet_activation_fn ActivationFn)
{
    if (ActivationFn == NULL)
    {
        ActivationFn = NN__StepFn;
    }

    assert(NN->LayerCount >= 1);
    assert(NN->InputCount == NN->Layers[0].InputCount);
    /* NOTE: NeuralNet_Create() guarantees that 
     * consecutive layers have the previous layer's output node count == the current layer's input node count */

    const float *X = NN->Inputs;
    for (int i = 0; i < NN->LayerCount; i++)
    {
        neuralnet_layer *Layer = &NN->Layers[i];

        NN__LinearCombination(
            Layer->NodeValues, 
            Layer->Weights, X, Layer->NodeBiases, 
            Layer->OutputCount, Layer->InputCount
        );

        /* NOTE: normalize outputs via activation fn ("squish" Y from -inf..+inf to 0..1) */
        for (int r = 0; r < Layer->OutputCount; r++)
        {
            Layer->NodeValues[r] = ActivationFn(Layer->NodeValues[r]);
        }

        X = Layer->NodeValues;
    }
}

void NeuralNet_BackPropagate(neuralnet *NN, neuralnet_backprop_config *Config)
{
    assert(NN->LayerCount >= 2);
    {
        /* update output delta */
        {
            const neuralnet_layer *OutputLayer = &NN->Layers[NN->LayerCount - 1];
            assert(OutputLayer->OutputCount == Config->ExpectedOutputCount);

            for (int i = 0; i < OutputLayer->OutputCount; i++)
            {
                float Y = OutputLayer->NodeValues[i];
                float Error = Y - Config->ExpectedOutputs[i];
                float Gradient = NN__SigmoidDerivativeY(Y); /* NOTE: derivative of sigmoid: y' = y(1 - y) */
                OutputLayer->DeltaWeight[i] = Error * Gradient;
                OutputLayer->DeltaBias[i] = Error * Gradient;
            }
        }

        /* update hidden layer delta */
        for (int i = NN->LayerCount - 2; i >= 0; i--)
        {
            neuralnet_layer *Curr = &NN->Layers[i];
            neuralnet_layer *Next = &NN->Layers[i + 1];

            for (int k = 0; k < Curr->OutputCount; k++)
            {
                float SumBias = 0, 
                      SumWeight = 0;
                for (int j = 0; j < Next->OutputCount; j++)
                {
                    SumBias += Curr->NodeBiases[j] * Next->DeltaBias[j];
                    SumWeight += Curr->Weights[j + k*Curr->InputCount] * Next->DeltaWeight[j];
                }
                float Gradient = NN__SigmoidDerivativeY(Curr->NodeValues[k]);
                Curr->DeltaWeight[k] = SumWeight * Gradient;
                Curr->DeltaBias[k] = SumBias * Gradient;
            }
        }

        /* update weights and biases */
        int InputCount = NN->InputCount;
        const float *Inputs = NN->Inputs;
        for (int i = 0; i < NN->LayerCount; i++)
        {
            neuralnet_layer *Curr = &NN->Layers[i];
            for (int k = 0; k < InputCount; k++)
            {
                for (int j = 0; j < Curr->OutputCount; j++)
                {
                    float DeltaWeight = Curr->DeltaWeight[j] * Inputs[k];
                    Curr->Weights[j + k*InputCount] -= DeltaWeight * Config->LearningRate;
                }

                float DeltaBias = Curr->DeltaBias[k] * 1.0;
                Curr->NodeBiases[k] -= DeltaBias * Config->LearningRate;
            }

            Inputs = Curr->NodeValues;
            InputCount = Curr->OutputCount;
        }
    }
}


static void *NN__DefaultAllocatorCallback(void *Data, neuralnet_allocator_param *Param)
{
    switch (Param->Mode)
    {
    case NNALLOC_ALLOCATE: return malloc(Param->Allocate.SizeBytes);
    case NNALLOC_FREE:     free(Param->Free.Ptr); break;
    }
    return NULL;
}

static void NN__Randomize(neuralnet *NN)
{
    srand(time(NULL));
    for (int n = 0; n < NN->LayerCount; n++)
    {
        neuralnet_layer *Layer = &NN->Layers[n];
        for (int k = 0; k < Layer->OutputCount; k++)
        {
            Layer->NodeBiases[k] = NN__GetRandomValue();
            Layer->NodeValues[k] = NN__GetRandomValue();
            for (int i = 0; i < Layer->InputCount; i++)
            {
                Layer->Weights[k*Layer->OutputCount + i] = NN__GetRandomValue();
            }
        }

    }
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

/* NOTE: Y = A . B^T, 
 * ColB is the number of columns in B (aka (B^T)^T) */
static void NN__MatDot(float *Y, const float *A, const float *BT, int RowA, int ColA, int RowBT)
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

static float NN__SigmoidDerivativeY(float Y)
{
    return Y*(1 - Y);
}

static float NN__GetRandomValue(void)
{
    return (float)rand() / RAND_MAX;
}

static float NN__StepFn(float Value)
{
    if (Value > 0.0)
        return 1.0;
    return 0;
}

#endif
