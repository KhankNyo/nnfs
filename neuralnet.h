#ifndef NEURALNET_H
#define NEURALNET_H

#include <stdbool.h>

typedef struct neuralnet neuralnet;
typedef struct neuralnet_layer neuralnet_layer;
typedef struct neuralnet_config neuralnet_config;
typedef struct neuralnet_backprop_config neuralnet_backprop_config;
typedef struct neuralnet_feedforward_config neuralnet_feedforward_config;
typedef struct neuralnet_allocator_param neuralnet_allocator_param;
typedef struct neuralnet_param_stats neuralnet_param_stats;
typedef enum 
{
    NNALLOC_ALLOCATE,
    NNALLOC_FREE,
} neuralnet_allocator_mode;
typedef float (*neuralnet_activation_fn)(float Value);
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
    float L2Lambda;
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

struct neuralnet_param_stats
{
    float WeightMin, WeightMax;
    float BiasMin, BiasMax;
};

/* memory allocation is not the focal point here, but since we're in C,
   it's kinda important to think about how you allocate memory */
struct neuralnet_allocator_param
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
};


neuralnet NeuralNet_Create(const neuralnet_config *Config);
neuralnet NeuralNet_CheapCopy(const neuralnet *NN);
void NeuralNet_Destroy(neuralnet *NN);

void NeuralNet_Randomize(neuralnet *NN);
void NeuralNet_FeedForward(neuralnet *NN, const neuralnet_feedforward_config *Config);
void NeuralNet_Backprop(neuralnet *NN, neuralnet_backprop_config *Config);
neuralnet_param_stats NeuralNet_GetParamStats(const neuralnet *NN);
float NeuralNet_CalcLoss(neuralnet *NN, const float *ExpectedOutputs, int OutputCount, float L2Lambda);

void NeuralNet_Print(const neuralnet *NN);
float *NeuralNet_GetOutput(neuralnet *NN);


struct neuralnet
{
    int InputCountB;
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
    int InputCountB;
    int InputCount;
    int OutputCount;
    /* [InputB x Output matrix] */
    float *Weights;

    /* [array with length of OutputCount] */
    float *Outputs;
    float *Deltas;
};


#endif /* NEURALNET_H */




#if defined(NEURALNET_IMPLEMENTATION) && !defined(NEURALNET_ALREADY_IMPLEMENTED)
#define NEURALNET_ALREADY_IMPLEMENTED


#include <stdlib.h> /* malloc, free */
#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
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

#define NN__SIMD_VEC_LEN 8
#define NN__MAX(a, b) ((a) > (b)? (a) : (b))
#define NN__MIN(a, b) ((a) < (b)? (a) : (b))

static void *NN__DefaultAllocatorCallback(void *Data, neuralnet_allocator_param *Param);
static float NN__DotProduct(const float *A, const float *B, int Length);
static void NN__MatMulABT(float *Y, const float *A, const float *BT, int RowA, int ColA, int RowBT);
static void NN__MatTranspose(float *Result, const float *Mat, int Row, int Col);
static void NN__MatSubInPlace(float *Lhs, const float *Rhs, int Row, int Col);
static void NN__MatScaleInPlace(float *Mat, float Scale, int Stride, int Row, int Col);
static float NN__GetRandomValue(void);
static float NN__Sigmoid(float Value);
static float NN__SigmoidDerivativeY(float Y);


neuralnet NeuralNet_Create(const neuralnet_config *Config)
{
    assert(Config->LayerCount >= 1 && "must have at leaast 1 layer (output layer)");
    neuralnet NN = { 
        .InputCount = Config->InputCount,
        .InputCountB = Config->InputCount + 1,
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
        NN.Inputs = NN__ALLOC(&NN, sizeof(NN.Inputs[0]) * NN.InputCountB);
        NN.Layers = NN__ALLOC(&NN, sizeof(NN.Layers[0]) * NN.LayerCount);
        int InputCount = NN.InputCount;
        int InputCountB = NN.InputCountB;
        int LargestSide = InputCountB;
        for (int i = 0; i < Config->LayerCount; i++)
        {
            int OutputCount = Config->NodeCountPerLayer[i];

            NN.Layers[i].Weights = NN__ALLOC(&NN, OutputCount*InputCountB*sizeof(NN.Layers[0].Weights[0]));
            NN.Layers[i].Deltas = NN__ALLOC(&NN, (OutputCount + 1)*sizeof(NN.Layers[0].Deltas[0]));
            NN.Layers[i].Outputs = NN__ALLOC(&NN, (OutputCount + 1)*sizeof(NN.Layers[0].Outputs[0]));
            NN.Layers[i].InputCount = InputCount;
            NN.Layers[i].InputCountB = InputCountB;
            NN.Layers[i].OutputCount = OutputCount;

            InputCount = OutputCount;
            InputCountB = OutputCount + 1;
            LargestSide = NN__MAX(OutputCount + 1, LargestSide);
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
        assert(Config->InputCount + 1 == NN->InputCountB);
        assert(Config->Inputs);
        memcpy(NN->Inputs, Config->Inputs, NN->InputCount * sizeof(NN->Inputs[0]));
        NN->Inputs[NN->InputCountB - 1] = 1.0;
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
        assert(X[Layer->InputCountB - 1] == 1.0);

        NN__MatMulABT(
            Layer->Outputs,
            Layer->Weights, X,
            Layer->InputCountB, Layer->OutputCount,
            1
        );

        /* NOTE: normalize outputs via activation fn ("squish" Y from -inf..+inf to 0..1) */
        for (int r = 0; r < Layer->OutputCount; r++)
        {
            Layer->Outputs[r] = ActivationFn(Layer->Outputs[r]);
        }

        X = Layer->Outputs;
    }
}

void NeuralNet_Backprop(neuralnet *NN, neuralnet_backprop_config *Config)
{
    /* deltas */
    {
        const neuralnet_layer *Last = NN->Layers + NN->LayerCount - 1;
        assert(Config->ExpectedOutputCount == Last->OutputCount);

        /* compute output layer deltas */
        for (int i = 0; i < Last->OutputCount; i++)
        {
            float Error = Last->Outputs[i] - Config->ExpectedOutputs[i];
            /* NOTE: hack, learning rate should be present during weight/bias update, not during delta calculation */
            Last->Deltas[i] = Config->LearningRate * Error * NN__SigmoidDerivativeY(Last->Outputs[i]);
        }

        /* compute hidden layer deltas */
        for (int i = NN->LayerCount - 2; i >= 0; i--)
        {
            neuralnet_layer *Next = NN->Layers + i + 1;
            neuralnet_layer *Curr = NN->Layers + i;

            /* TODO: benchmark transpose, because it is not cache friendly */
            NN__MatTranspose(NN->ScratchMatrix, Next->Weights, Next->InputCountB, Next->OutputCount);
            NN__MatMulABT(
                Curr->Deltas, 
                NN->ScratchMatrix, Next->Deltas, 
                Next->OutputCount, Next->InputCountB, 1
            );
            /* NOTE: Next->InputCountB includes node with value 1.0 for bias, Curr->OutputCount does not */
            for (int k = 0; k < Next->InputCountB; k++)
            {
                float Tmp = Config->LearningRate * NN__SigmoidDerivativeY(Curr->Outputs[k]);
                Curr->Deltas[k] *= Tmp;
            }
        }
    }

    float L2Regularization = 1.0 - Config->L2Lambda;

    /* update weights and biases */
    int InputCount = NN->InputCount;
    int InputCountB = NN->InputCountB;
    float *Inputs = NN->Inputs;
    for (int i = 0; i < NN->LayerCount; i++)
    {
        neuralnet_layer *Curr = NN->Layers + i;
        NN__MatMulABT(NN->ScratchMatrix, Curr->Deltas, Inputs, 1, Curr->OutputCount, InputCountB);
        /* NOTE: scaling the weights will not affect biases because InputCountB is the stride, InputCount is the row length */
        NN__MatScaleInPlace(Curr->Weights, L2Regularization, InputCountB, InputCount, Curr->OutputCount);
        NN__MatSubInPlace(Curr->Weights, NN->ScratchMatrix, InputCountB, Curr->OutputCount);

        Inputs = Curr->Outputs;
        InputCount = Curr->OutputCount;
        InputCountB = Curr->OutputCount + 1;
    }
}

neuralnet_param_stats NeuralNet_GetParamStats(const neuralnet *NN)
{
    neuralnet_param_stats Stats = { 
        .BiasMax = -FLT_MAX,
        .BiasMin = FLT_MAX,
        .WeightMax = -FLT_MAX,
        .WeightMin = FLT_MAX,
    };
    for (int i = 0; i < NN->LayerCount; i++)
    {
        neuralnet_layer *Layer = NN->Layers + i;
        for (int k = 0; k < Layer->OutputCount; k++)
        {
            int Col = k*Layer->InputCountB;
            for (int j = 0; j < Layer->InputCount; j++)
            {
                int Index = Col + j;
                Stats.WeightMax = NN__MAX(Stats.WeightMax, Layer->Weights[Index]);
                Stats.WeightMin = NN__MIN(Stats.WeightMin, Layer->Weights[Index]);
            }

            Stats.BiasMax = NN__MAX(Stats.BiasMax, Layer->Weights[Col + Layer->InputCountB - 1]);
            Stats.BiasMin = NN__MIN(Stats.BiasMin, Layer->Weights[Col + Layer->InputCountB - 1]);
        }
    }
    return Stats;
}


/* MSE + L2 regularization */
float NeuralNet_CalcLoss(neuralnet *NN, const float *ExpectedOutputs, int OutputCount, float L2Lambda)
{
    float Sum = 0;
    const float *Outputs = NeuralNet_GetOutput(NN);
    for (int i = 0; i < OutputCount; i++)
    {
        float Tmp = (ExpectedOutputs[i] - Outputs[i]);
        Sum += Tmp*Tmp;
    }

    float L2 = 0;
    for (int i = 0; i < NN->LayerCount; i++)
    {
        float Sum = 0;
        /* NOTE: convoluted code to add the squares of weights,
         * but doing it this way helped the compiler vectorize the code
         * without having to write any simd intrinsics (runtime halved) */
        {
            neuralnet_layer *Layer = NN->Layers + i;
            for (int h = 0; h < Layer->OutputCount; h++)
            {
                float *WeightPtr = Layer->Weights + h*Layer->InputCountB;
                float Weights[NN__SIMD_VEC_LEN] = { 0 };
                for (int k = 0; k < Layer->InputCount / NN__SIMD_VEC_LEN; k++)
                {
                    for (int j = 0; j < NN__SIMD_VEC_LEN; j++)
                    {
                        float Weight = *WeightPtr++;
                        Weights[j] += Weight*Weight;
                    }
                }
                for (int k = 0; k < NN__SIMD_VEC_LEN; k++)
                {
                    Sum += Weights[k];
                }
                for (int k = 0; k < Layer->InputCount % NN__SIMD_VEC_LEN; k++)
                {
                    float Weight = *WeightPtr++;
                    Sum += Weight*Weight;
                }
            }
        }
        L2 += Sum;
    }
    return 0.5 * Sum + 0.5 * L2 * L2Lambda;
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
        for (int k = 0; k < Layer->OutputCount + 1; k++)
            printf("%6.3f ", Layer->Outputs[k]);
        printf("]\n");

        printf("        node delta: [ ");
        for (int k = 0; k < Layer->OutputCount; k++)
            printf("%6.3f ", Layer->Deltas[k]);
        printf("]\n");

        printf("        weights:\n");
        for (int k = 0; k < Layer->OutputCount; k++)
        {
            printf("            [ ");
            for (int j = 0; j < Layer->InputCountB; j++)
            {
                printf("%6.3f ", Layer->Weights[j + k*Layer->InputCountB]);
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
    /* NOTE: for biases */
    NN->Inputs[NN->InputCountB - 1] = 1.0;
    for (int n = 0; n < NN->LayerCount; n++)
    {
        neuralnet_layer *Layer = &NN->Layers[n];
        for (int k = 0; k < Layer->OutputCount; k++)
        {
            Layer->Outputs[k] = NN__GetRandomValue();
            for (int i = 0; i < Layer->InputCountB; i++)
            {
                Layer->Weights[k*Layer->InputCountB + i] = NN__GetRandomValue();
            }
        }
        /* NOTE: for biases */
        Layer->Outputs[Layer->OutputCount] = 1.0;
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


#if defined(NEURALNET_USE_SIMD)

#include <x86intrin.h>

static float NN__DotProduct(const float *A, const float *B, int Length)
{
    __m256 Accum = _mm256_setzero_ps();
    for (int i = 0; i < Length / NN__SIMD_VEC_LEN; i++)
    {
        __m256 VecA = _mm256_loadu_ps(A);
        __m256 VecB = _mm256_loadu_ps(B);
        Accum = _mm256_fmadd_ps(VecA, VecB, Accum); /* Accum += A * B; */
        A += NN__SIMD_VEC_LEN;
        B += NN__SIMD_VEC_LEN;
    }

    float Result = 0;
    {
        /* Accum[3, 2, 1, 0] -> Tmp[2, 3, 0, 1] */
        __m256 Tmp = _mm256_shuffle_ps(Accum, Accum, _MM_SHUFFLE(2, 3, 0, 1));
        /* Accum[23, 23, 01, 01] <- Accum[3, 2, 1, 0] + Tmp[2, 3, 0, 1] */
        Accum = _mm256_add_ps(Accum, Tmp);
        /* Accum[23, 23, 01, 01] -> Tmp[01, 01, 23, 23] */
        Tmp = _mm256_shuffle_ps(Accum, Accum, _MM_SHUFFLE(0, 0, 2, 2));
        /* Accum[0123, 0123, 0123, 0123] <- Accum[23, 23, 01, 01] + Tmp[01, 01, 23, 23] */
        Accum = _mm256_add_ps(Accum, Tmp);

        __m128 Low = _mm256_extractf128_ps(Accum, 0);
        __m128 High = _mm256_extractf128_ps(Accum, 1);
        __m128 Accum128 = _mm_add_ps(Low, High);
        _mm_store_ss(&Result, Accum128);
    }

    for (int i = 0; i < Length % NN__SIMD_VEC_LEN; i++)
    {
        Result += *A * *B;
        A++;
        B++;
    }
    return Result;
}

static void NN__MatSubInPlace(float *Lhs, const float *Rhs, int Row, int Col)
{
    int Length = Col*Row;

    /* main */
    for (int i = 0; i < Length / NN__SIMD_VEC_LEN; i++)
    {
        __m256 Result = _mm256_sub_ps(
            _mm256_loadu_ps(Lhs), 
            _mm256_loadu_ps(Rhs)
        );
        _mm256_storeu_ps(Lhs, Result);

        Lhs += NN__SIMD_VEC_LEN;
        Rhs += NN__SIMD_VEC_LEN;
    }

    /* residue */
    for (int i = 0; i < Length % NN__SIMD_VEC_LEN; i++)
    {
        float Result = *Lhs - *Rhs;
        *Lhs = Result;

        Lhs++;
        Rhs++;
    }
}

#else

static float NN__DotProduct(const float *A, const float *B, int Length)
{
    float Result = 0;
    for (int i = 0; i < Length; i++)
    {
        Result += A[i] * B[i];
    }
    return Result;
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
#endif


static void NN__MatScaleInPlace(float *Mat, float Scale, int Stride, int Row, int Col)
{
    for (int y = 0; y < Col; y++)
    {
        /* compiler was able to vectorize the code with -O3 */
        for (int x = 0; x < Row; x++)
        {
            int Index = y*Stride + x;
            Mat[Index] *= Scale;
        }
    }
}

/* NOTE: Y = A . B^T */
static void NN__MatMulABT(float *Y, const float *A, const float *BT, int RowA, int ColA, int RowBT)
{
    for (int Ca = 0; Ca < ColA; Ca++)
    {
        for (int Rtb = 0; Rtb < RowBT; Rtb++)
        {
            const float *RowMatA = A + Ca*RowA;
            const float *ColMatB = BT + Rtb*RowA;
            float Dp = NN__DotProduct(RowMatA, ColMatB, RowA);
            Y[Ca*RowBT + Rtb] = Dp;
        }
    }
}

static void NN__MatTranspose(float *Result, const float *Mat, int Row, int Col)
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
    return (float)rand() / RAND_MAX * 2.0 - 1;
}

static float NN__Sigmoid(float Value)
{
    return 1.0 / (1.0 + expf(-Value));
}

#endif
