#ifndef NEURALNET_H
#define NEURALNET_H

#include <stdbool.h>

typedef struct neuralnet neuralnet;
typedef struct neuralnet_layer neuralnet_layer;
typedef struct neuralnet_config neuralnet_config;
typedef struct neuralnet_training_config neuralnet_training_config;
typedef struct neuralnet_graddesc_config neuralnet_graddesc_config;
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

struct neuralnet_training_config
{
    int SetCount;
    int InputCount;
    int ExpectedOutputCount;

    const float *Inputs;
    const float *ExpectedOutputs;

    neuralnet_activation_fn ActivationFn;
};

struct neuralnet_graddesc_config
{
    float Rate;
    float Epsilon;
    int Iterations;
    const neuralnet_training_config *TrainingConfig;
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
};



neuralnet NeuralNet_Create(const neuralnet_config *Config);
neuralnet NeuralNet_CheapCopy(const neuralnet *NN);
void NeuralNet_Destroy(neuralnet *NN);

void NeuralNet_Randomize(neuralnet *NN);
void NeuralNet_FeedForward(neuralnet *NN, neuralnet_activation_fn Fn);
/* NOTE: to use backprop, activation function used for feed forward must be differentiable (no step fn, ex: RELU) */
float NeuralNet_Mse(neuralnet *NN, const neuralnet_training_config *Config);
void NeuralNet_Backprop(neuralnet *NN, const float *Errors, float LearningRate);

void NeuralNet_Print(const neuralnet *NN);
float *NeuralNet_GetOutput(neuralnet *NN);



#endif /* NEURALNET_H */




#if defined(NEURALNET_IMPLEMENTATION) && !defined(NEURALNET_ALREADY_IMPLEMENTED)
#define NEURALNET_ALREADY_IMPLEMENTED


#include <stdlib.h> /* malloc, free */
#include <stdint.h>
#include <assert.h>
#include <time.h>
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

static void *NN__DefaultAllocatorCallback(void *Data, neuralnet_allocator_param *Param);
static void NN__LinearCombination(float *Y, const float *M, const float *X, const float *B, int Row, int Col);
static void NN__MatDot(float *Y, const float *A, const float *BT, int RowA, int ColA, int RowBT);
static void NN__MatTranpose(float *Result, const float *Mat, int Row, int Col);
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
        for (int i = 0; i < Config->LayerCount; i++)
        {
            int OutputCount = Config->NodeCountPerLayer[i];

            NN.Layers[i].Weights = NN__ALLOC(&NN, OutputCount*InputCount*sizeof(NN.Layers[0].Weights[0]));
            NN.Layers[i].NodeBiases = NN__ALLOC(&NN, OutputCount*sizeof(NN.Layers[0].NodeBiases[0]));
            NN.Layers[i].NodeValues = NN__ALLOC(&NN, OutputCount*sizeof(NN.Layers[0].NodeValues[0]));
            NN.Layers[i].InputCount = InputCount;
            NN.Layers[i].OutputCount = OutputCount;

            InputCount = OutputCount;
        }
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
        NN__FREE(NN, NN->Layers[i].NodeBiases);
        NN__FREE(NN, NN->Layers[i].NodeValues);
    }
    NN__FREE(NN, NN->Layers);
    NN__FREE(NN, NN->Inputs);
}

void NeuralNet_FeedForward(neuralnet *NN, neuralnet_activation_fn ActivationFn)
{
    if (ActivationFn == NULL)
    {
        ActivationFn = NN__Sigmoid;
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

float NeuralNet_Mse(neuralnet *NN, const neuralnet_training_config *Config)
{
    assert(Config->InputCount == NN->InputCount);
    assert(Config->ExpectedOutputCount == NN->Layers[NN->LayerCount - 1].OutputCount);

    float Result = 0;
    for (int i = 0; i < Config->SetCount; i++)
    {
        const float *Inputs = Config->Inputs + Config->InputCount*i;
        const float *Expected = Config->ExpectedOutputs + Config->ExpectedOutputCount*i;

        memcpy(NN->Inputs, Inputs, NN->InputCount * sizeof(NN->Inputs[0]));
        NeuralNet_FeedForward(NN, Config->ActivationFn);
        const float *Predictions = NeuralNet_GetOutput(NN);

        for (int k = 0; k < Config->ExpectedOutputCount; k++)
        {
            float Diff = Predictions[k] - Expected[k];
            Result += Diff*Diff;
        }
    }
    return Result / Config->SetCount;
}

static void NN__FiniteDiff(neuralnet *OutGradient, neuralnet *Model, float Epsilon, const neuralnet_training_config *MseConfig)
{
    float Loss = NeuralNet_Mse(Model, MseConfig);
    for (int i = 0; i < Model->LayerCount; i++)
    {
        neuralnet_layer *Layer = &Model->Layers[i];
        for (int k = 0; k < Layer->OutputCount; k++)
        {
            for (int n = 0; n < Layer->InputCount; n++)
            {
                /* weights */
                int Index = n + k*Layer->OutputCount;

                float Tmp = Layer->Weights[Index];
                Layer->Weights[Index] += Epsilon;
                OutGradient->Layers[i].Weights[Index] = (NeuralNet_Mse(Model, MseConfig) - Loss) / Epsilon;
                Layer->Weights[Index] = Tmp;
            }

            /* biases */
            float Tmp = Layer->NodeBiases[k];
            Layer->NodeBiases[k] += Epsilon;
            OutGradient->Layers[i].NodeBiases[k] = (NeuralNet_Mse(Model, MseConfig) - Loss) / Epsilon;
            Layer->NodeBiases[k] = Tmp;
        }
    }
}

void NeuralNet_GradientDescent(neuralnet *NN, const neuralnet_graddesc_config *Config)
{
    neuralnet Gradient = NeuralNet_CheapCopy(NN);
    for (int i = 0; i < Config->Iterations; i++)
    {
        NN__FiniteDiff(&Gradient, NN, Config->Epsilon, Config->TrainingConfig);
        for (int i = 0; i < NN->LayerCount; i++)
        {
            neuralnet_layer *Layer = &NN->Layers[i];
            for (int k = 0; k < Layer->OutputCount; k++)
            {
                for (int n = 0; n < Layer->InputCount; n++)
                {
                    /* weights */
                    int Index = n + k*Layer->OutputCount;
                    Layer->Weights[Index] -= Config->Rate * Gradient.Layers[i].Weights[Index];
                }

                /* biases */
                Layer->NodeBiases[k] -= Config->Rate * Gradient.Layers[i].NodeBiases[k];
            }
        }
    }
    NeuralNet_Destroy(&Gradient);
}

void NeuralNet_Backprop(neuralnet *NN, const float *Errors, float LearningRate)
{

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
            printf("%6.3f ", Layer->NodeValues[k]);
        printf("]\n");

        printf("        node bias:  [ ");
        for (int k = 0; k < Layer->OutputCount; k++)
            printf("%6.3f ", Layer->NodeBiases[k]);
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
        printf("%g ", NN->Layers[NN->LayerCount - 1].NodeValues[i]);
    printf("]\n");
}

void NeuralNet_Randomize(neuralnet *NN)
{
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

float *NeuralNet_GetOutput(neuralnet *NN)
{
    return NN->Layers[NN->LayerCount - 1].NodeValues;
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

static float NN__Sigmoid(float Value)
{
    return 1.0 / (1.0 + expf(-Value));
}

#endif
