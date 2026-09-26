#ifndef NNFXP_H
#define NNFXP_H

/* this can be used independently from neuralnet.h */
/* TODO: multiply-accumulate in wider type */

#include <stdbool.h>
#include <stdint.h>

#ifndef nnfxp_type
#  define NNFXP_TYPE_MAX INT32_MAX
#  define NNFXP_TYPE_MIN INT32_MIN
#  define nnfxp_type int32_t
#  define nnfxp_utype uint32_t
#endif /* nnfxp_type */

#ifndef NNFXP_FRACTION_BITS
#  define NNFXP_FRACTION_BITS 13
#endif /* NNFXP_FRACTION_BITS */




typedef struct nnfxp nnfxp;
typedef struct nnfxp_layer nnfxp_layer;
typedef struct nnfxp_feedforward_config nnfxp_feedforward_config;
typedef struct nnfxp_allocator_param nnfxp_allocator_param;
typedef struct nnfxp_config nnfxp_config;
typedef struct nnfxp_backprop_config nnfxp_backprop_config;
typedef struct nnfxp_param_stats nnfxp_param_stats;
typedef enum
{
    NNFXP_ALLOCATE,
    NNFXP_FREE,
} nnfxp_allocator_mode;
typedef void *(*nnfxp_allocator_callback)(void *UserData, nnfxp_allocator_param *Param);
typedef nnfxp_type (*nnfxp_activation_callback)(void *UserData, nnfxp_type X);
typedef nnfxp_type (*nnfxp_rand_callback)(void *UserData);

struct nnfxp_allocator_param
{
    nnfxp_allocator_mode Mode;
    union {
        struct {
            int SizeBytes;
        } Allocate;
        struct {
            void *Ptr;
        } Free;
    };
};

struct nnfxp_param_stats
{
    nnfxp_type WeightMin, WeightMax;
    nnfxp_type BiasMin, BiasMax;
};

struct nnfxp_config
{
    int InputCount;
    int LayerCount;
    int *NodeCountPerLayer;

    void *AllocatorData;
    nnfxp_allocator_callback AllocatorCallback;
    void *RandData;
    nnfxp_rand_callback RandCallback;
    void *ActivationData;
    nnfxp_activation_callback ActivationCallback;
};

struct nnfxp_feedforward_config
{
    int InputCount;
    const nnfxp_type *Inputs;
};

struct nnfxp_backprop_config
{
    nnfxp_type L2Lambda;
    nnfxp_type LearningRate;
    int ExpectedOutputCount;
    const nnfxp_type *ExpectedOutputs;
};


// TODO:
// nnfxp Nnfxp_CreateFromNeuralNet(const neuralnet *NN)
void Nnfxp_Create(nnfxp *NN, const nnfxp_config *Config);
void Nnfxp_Destroy(nnfxp *NN);

void Nnfxp_Randomize(nnfxp *NN);
void Nnfxp_FeedForward(nnfxp *NN, const nnfxp_feedforward_config *Config);
void Nnfxp_Backprop(nnfxp *NN, const nnfxp_backprop_config *Config);
nnfxp_type Nnfxp_CalcLoss(nnfxp *NN, const nnfxp_type *ExpectedOutputs, int OutputCount, nnfxp_type L2Lambda);

nnfxp_type *Nnfxp_GetOutputs(nnfxp *NN);
void Nnfxp_Print(nnfxp *NN);


#define NNFXP_FLT_EXTRACT_FRAC(val) ((val) - (float)(nnfxp_utype)(val))

#define NNFXP_EXTRACT_FRAC(fxp) ((fxp) & (NNFXP_ONE - 1))
#define NNFXP_EXTRACT_INT(fxp) ((fxp) >> NNFXP_FRACTION_BITS)

#define NNFXP_ONE ((nnfxp_type)1 << NNFXP_FRACTION_BITS)
#define NNFXP(constant) (nnfxp_type)((constant) * NNFXP_ONE)
#define NNFXP_FLT(fxp) ((float)(fxp) / NNFXP_ONE)

#define NNFXP_ADD(a, b) ((a) + (b))
#define NNFXP_SUB(a, b) ((a) - (b))
#define NNFXP_ADDI(x, i) NNFXP_ADD(x, (i)*NNFXP_ONE)
#define NNFXP_SUBI(x, i) NNFXP_SUB(x, (i)*NNFXP_ONE)
#define NNFXP_ADDC(x, constant) NNFXP_ADD(x, NNFXP(constant)) 
#define NNFXP_SUBC(x, constant) NNFXP_SUB(x, NNFXP(constant))

#define NNFXP_MUL(a, b) (((a) * (b)) >> (NNFXP_FRACTION_BITS))
#define NNFXP_DIV(a, b) (((a) / (b)) << NNFXP_FRACTION_BITS)
#define NNFXP_DIVL(a, b) ((((a) << NNFXP_FRACTION_BITS) / (b)))
#define NNFXP_DIVR(a, b) (((a) / ((b) >> NNFXP_FRACTION_BITS)))
#define NNFXP_MULI(x, i) ((x) * (i))
#define NNFXP_DIVI(x, i) ((x) / (i))
#define NNFXP_MULC(x, constant) NNFXP_MUL(x, NNFXP(constant))
#define NNFXP_DIVC(x, constant) NNFXP_DIV(x, NNFXP(constant))




struct nnfxp_layer
{
    int InputCount;
    int InputCountB;
    int OutputCount;
    /* InputCountB x OutputCount */
    nnfxp_type *Weights;

    /* OutputCount */
    nnfxp_type *Outputs;
    nnfxp_type *Deltas;
};

struct nnfxp
{
    int InputCount;
    int InputCountB;
    int LayerCount;
    nnfxp_type *Inputs;
    nnfxp_layer *Layers;
    nnfxp_type *ScratchMatrix;

    void *AllocatorData;
    nnfxp_allocator_callback AllocatorCallback;
    void *RandData;
    nnfxp_rand_callback RandCallback;
    void *ActivationData;
    nnfxp_activation_callback ActivationCallback;
};


#endif /* NNFXP */


#if defined(NNFXP_IMPLEMENTATION) && !defined(NNFXP_ALREADY_IMPLEMENTED)
#define NNFXP_ALREADY_IMPLEMENTED

#include <stdlib.h> /* malloc/free */
#include <string.h> /* memcpy */
#include <stdio.h> /* printf */
#include <assert.h>


#define NNFXP__ALLOC(p_nn, size_bytes) \
    (p_nn)->AllocatorCallback(\
        (p_nn)->AllocatorData, \
        &(nnfxp_allocator_param) { \
            .Mode = NNFXP_ALLOCATE, \
            .Allocate.SizeBytes = (size_bytes)\
        }\
    )
#define NNFXP__FREE(p_nn, ptr) \
    (p_nn)->AllocatorCallback(\
        (p_nn)->AllocatorData,\
        &(nnfxp_allocator_param) {\
            .Mode = NNFXP_FREE, \
            .Free.Ptr = (ptr), \
        }\
    )
#define NNFXP__RAND(p_nn) (p_nn)->RandCallback((p_nn)->RandData)
#define NNFXP__IN_RANGE(lower, n, upper) ((lower) <= (n) && (n) <= (upper))

#define NNFXP__SIMD_VEC_LEN 8
#define NNFXP__MAX(a, b) ((a) > (b)? (a) : (b))
#define NNFXP__MIN(a, b) ((a) < (b)? (a) : (b))



static void *Nnfxp__DefaultAllocatorCallback(void *Data, nnfxp_allocator_param *Param)
{
    (void)Data;
    switch (Param->Mode)
    {
    case NNFXP_ALLOCATE: return malloc(Param->Allocate.SizeBytes);
    case NNFXP_FREE:     free(Param->Free.Ptr); break;
    }
    return NULL;
}

static nnfxp_type Nnfxp__DefaultRandCallback(void *Data)
{
    (void)Data;
    float Result = (float)rand() / RAND_MAX * 2.0 - 1;
    return NNFXP(Result);
}

static nnfxp_type Nnfxp__Sigmoid(void *Data, nnfxp_type X)
{
    (void)Data;
    /* piecewise approx of 1/(1 + e^-x) */
    if (NNFXP__IN_RANGE(NNFXP(-1.0), X, NNFXP(1.0)))
    {
        return NNFXP(0.5) + NNFXP_MUL(NNFXP(0.235), X);
    }
    else if (NNFXP__IN_RANGE(NNFXP(-1.5), X, NNFXP(-1.0)))
    {
        return NNFXP(0.445) + NNFXP_MUL(NNFXP(0.175), X);
    }
    else if (NNFXP__IN_RANGE(NNFXP(1.0), X, NNFXP(1.5)))
    {
        return NNFXP(0.555) + NNFXP_MUL(NNFXP(0.175), X);
    }
    else if (NNFXP__IN_RANGE(NNFXP(-2.0), X, NNFXP(-1.5)))
    {
        return NNFXP(0.3708) + NNFXP_MUL(NNFXP(0.126), X);
    }
    else if (NNFXP__IN_RANGE(NNFXP(1.5), X, NNFXP(2.0)))
    {
        return NNFXP(0.6308) + NNFXP_MUL(NNFXP(0.126), X);
    }
    else if (NNFXP__IN_RANGE(NNFXP(-3.0), X, NNFXP(-2.0)))
    {
        return NNFXP(0.26) + NNFXP_MUL(NNFXP(0.07178), X);
    }
    else if (NNFXP__IN_RANGE(NNFXP(2.0), X, NNFXP(3.0)))
    {
        return NNFXP(0.74) + NNFXP_MUL(NNFXP(0.07178), X);
    }
    else if (NNFXP__IN_RANGE(NNFXP(-5.0), X, NNFXP(-3.0)))
    {
        return NNFXP(0.105) + NNFXP_MUL(NNFXP(0.0203665), X);
    }
    else if (NNFXP__IN_RANGE(NNFXP(3.0), X, NNFXP(5.0)))
    {
        return NNFXP(0.895) + NNFXP_MUL(NNFXP(0.0203665), X);
    }
    else if (X > NNFXP(5))
    {
        return NNFXP_ONE;
    }
    return 0;
}

static nnfxp_type Nnfxp__SigmoidDerivativeY(nnfxp_type Y)
{
    return NNFXP_MUL(Y, (NNFXP_ONE - Y));
}

static nnfxp_type Nnfxp__DotProduct(const nnfxp_type *A, const nnfxp_type *B, int Length)
{
    nnfxp_type Result = 0;
    for (int i = 0; i < Length; i++)
    {
        Result += NNFXP_MUL(A[i], B[i]);
    }
    return Result;
}

/* NOTE: Out = A*B^T */
static void Nnfxp__MatMulABT(nnfxp_type *Out, const nnfxp_type *A, const nnfxp_type *BT, int RowA, int ColA, int RowBT)
{
    for (int Ca = 0; Ca < ColA; Ca++)
    {
        for (int Rbt = 0; Rbt < RowBT; Rbt++)
        {
            const nnfxp_type *RowMatA = A + Ca*RowA;
            const nnfxp_type *ColMatB = BT + Rbt*RowA;
            nnfxp_type Dp = Nnfxp__DotProduct(RowMatA, ColMatB, RowA);
            Out[Ca*RowBT + Rbt] = Dp;
        }
    }
}

static void Nnfxp__MatScaleInPlace(nnfxp_type *Mat, nnfxp_type Scale, int Stride, int Row, int Col)
{
    for (int y = 0; y < Col; y++)
    {
        /* compiler was able to vectorize the code with -O3 */
        for (int x = 0; x < Row; x++)
        {
            int Index = y*Stride + x;
            Mat[Index] = NNFXP_MUL(Mat[Index], Scale);
        }
    }
}

static void Nnfxp__MatSubInPlace(nnfxp_type *Lhs, nnfxp_type *Rhs, int Row, int Col)
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

static void Nnfxp__MatTranspose(nnfxp_type *Result, const nnfxp_type *Mat, int Row, int Col)
{
    for (int c = 0; c < Col; c++)
    {
        for (int r = 0; r < Row; r++)
        {
            Result[r*Col + c] = Mat[r + c*Row];
        }
    }
}




void Nnfxp_Create(nnfxp *NN, const nnfxp_config *Config)
{
    assert(Config->LayerCount >= 1 && "must have at leaast 1 layer (output layer)");
    *NN = (nnfxp) { 
        .InputCount = Config->InputCount,
        .InputCountB = Config->InputCount + 1,
        .LayerCount = Config->LayerCount,
    };
    if (Config->AllocatorCallback != NULL)
    {
        NN->AllocatorCallback = Config->AllocatorCallback;
        NN->AllocatorData = Config->AllocatorData;
    }
    else
    {
        NN->AllocatorCallback = Nnfxp__DefaultAllocatorCallback;
        NN->AllocatorData = NULL;
    }
    if (Config->RandCallback != NULL)
    {
        NN->RandCallback = Config->RandCallback;
        NN->RandData = Config->RandData;
    }
    else
    {
        NN->RandCallback = Nnfxp__DefaultRandCallback;
        NN->RandData = NULL;
    }
    if (Config->ActivationCallback != NULL)
    {
        NN->ActivationCallback = Config->ActivationCallback;
        NN->ActivationData = Config->ActivationData;
    }
    else
    {
        NN->ActivationCallback = Nnfxp__Sigmoid;
        NN->ActivationData = NULL;
    }

    /* allocate needed mem */
    {
        NN->Inputs = NNFXP__ALLOC(NN, sizeof(NN->Inputs[0]) * NN->InputCountB);
        NN->Layers = NNFXP__ALLOC(NN, sizeof(NN->Layers[0]) * NN->LayerCount);
        int InputCount = NN->InputCount;
        int InputCountB = NN->InputCountB;
        int LargestSide = InputCountB;
        for (int i = 0; i < Config->LayerCount; i++)
        {
            int OutputCount = Config->NodeCountPerLayer[i];

            NN->Layers[i].Weights = NNFXP__ALLOC(NN, OutputCount*InputCountB*sizeof(NN->Layers[0].Weights[0]));
            NN->Layers[i].Deltas = NNFXP__ALLOC(NN, (OutputCount + 1)*sizeof(NN->Layers[0].Deltas[0]));
            NN->Layers[i].Outputs = NNFXP__ALLOC(NN, (OutputCount + 1)*sizeof(NN->Layers[0].Outputs[0]));
            NN->Layers[i].InputCount = InputCount;
            NN->Layers[i].InputCountB = InputCountB;
            NN->Layers[i].OutputCount = OutputCount;

            InputCount = OutputCount;
            InputCountB = OutputCount + 1;
            LargestSide = NNFXP__MAX(OutputCount + 1, LargestSide);
        }
        NN->ScratchMatrix = NNFXP__ALLOC(NN, LargestSide*LargestSide*sizeof(NN->ScratchMatrix[0]));
    }

    Nnfxp_Randomize(NN);
}

void Nnfxp_Destroy(nnfxp *NN)
{
    for (int i = 0; i < NN->LayerCount; i++)
    {
        NNFXP__FREE(NN, NN->Layers[i].Weights);
        NNFXP__FREE(NN, NN->Layers[i].Deltas);
        NNFXP__FREE(NN, NN->Layers[i].Outputs);
    }
    NNFXP__FREE(NN, NN->Layers);
    NNFXP__FREE(NN, NN->Inputs);
    NNFXP__FREE(NN, NN->ScratchMatrix);
}

void Nnfxp_Randomize(nnfxp *NN)
{
    NN->Inputs[NN->InputCountB - 1] = NNFXP_ONE;
    for (int i = 0; i < NN->LayerCount; i++)
    {
        nnfxp_layer *Layer = NN->Layers + i;
        for (int k = 0; k < Layer->OutputCount; k++)
        {
            Layer->Outputs[k] = NNFXP__RAND(NN);
            for (int j = 0; j < Layer->InputCountB; j++)
            {
                Layer->Weights[k*Layer->InputCountB + j] = NNFXP__RAND(NN);
            }
        }
        /* NOTE: for biases */
        Layer->Outputs[Layer->OutputCount] = NNFXP_ONE;
    }
}

void Nnfxp_FeedForward(nnfxp *NN, const nnfxp_feedforward_config *Config)
{
    assert(Config->Inputs && Config->InputCount == NN->InputCount);
    memcpy(NN->Inputs, Config->Inputs, Config->InputCount*sizeof(NN->Inputs[0]));
    NN->Inputs[NN->InputCountB - 1] = NNFXP_ONE;
    nnfxp_activation_callback ActivationFn = NN->ActivationCallback;
    assert(ActivationFn);

    const nnfxp_type *X = NN->Inputs;
    for (int i = 0; i < NN->LayerCount; i++)
    {
        nnfxp_layer *Layer = &NN->Layers[i];
        assert(X[Layer->InputCountB - 1] == NNFXP_ONE);

        Nnfxp__MatMulABT(
            Layer->Outputs, Layer->Weights, X,
            Layer->InputCountB, Layer->OutputCount, 1
        );

        /* NOTE: normalize outputs via activation fn ("squish" Y from -inf..+inf to 0..1) */
        for (int r = 0; r < Layer->OutputCount; r++)
        {
            Layer->Outputs[r] = ActivationFn(NN->ActivationData, Layer->Outputs[r]);
        }

        X = Layer->Outputs;
    }
}

void Nnfxp_Backprop(nnfxp *NN, const nnfxp_backprop_config *Config)
{
    /* deltas */
    {
        const nnfxp_layer *Last = NN->Layers + NN->LayerCount - 1;
        assert(Config->ExpectedOutputCount == Last->OutputCount);

        /* compute output layer deltas */
        for (int i = 0; i < Last->OutputCount; i++)
        {
            nnfxp_type Error = Last->Outputs[i] - Config->ExpectedOutputs[i];
            /* NOTE: hack, learning rate should be present during weight/bias update, not during delta calculation */
            Last->Deltas[i] = NNFXP_MUL(NNFXP_MUL(Config->LearningRate, Error), Nnfxp__SigmoidDerivativeY(Last->Outputs[i]));
        }

        /* compute hidden layer deltas */
        for (int i = NN->LayerCount - 2; i >= 0; i--)
        {
            nnfxp_layer *Next = NN->Layers + i + 1;
            nnfxp_layer *Curr = NN->Layers + i;

            /* TODO: benchmark transpose, because it is not cache friendly */
            Nnfxp__MatTranspose(NN->ScratchMatrix, Next->Weights, Next->InputCountB, Next->OutputCount);
            Nnfxp__MatMulABT(
                Curr->Deltas, 
                NN->ScratchMatrix, Next->Deltas, 
                Next->OutputCount, Next->InputCountB, 1
            );
            /* NOTE: Next->InputCountB includes node with value 1.0 for bias, Curr->OutputCount does not */
            for (int k = 0; k < Next->InputCountB; k++)
            {
                nnfxp_type Tmp = NNFXP_MUL(Config->LearningRate, Nnfxp__SigmoidDerivativeY(Curr->Outputs[k]));
                Curr->Deltas[k] = NNFXP_MUL(Curr->Deltas[k], Tmp);
            }
        }
    }

    nnfxp_type L2Regularization = NNFXP_ONE - Config->L2Lambda;

    /* update weights and biases */
    int InputCount = NN->InputCount;
    int InputCountB = NN->InputCountB;
    nnfxp_type *Inputs = NN->Inputs;
    for (int i = 0; i < NN->LayerCount; i++)
    {
        nnfxp_layer *Curr = NN->Layers + i;
        Nnfxp__MatMulABT(NN->ScratchMatrix, Curr->Deltas, Inputs, 1, Curr->OutputCount, InputCountB);
        /* NOTE: scaling the weights will not affect biases because InputCountB is the stride, InputCount is the row length */
        Nnfxp__MatScaleInPlace(Curr->Weights, L2Regularization, InputCountB, InputCount, Curr->OutputCount);
        Nnfxp__MatSubInPlace(Curr->Weights, NN->ScratchMatrix, InputCountB, Curr->OutputCount);

        Inputs = Curr->Outputs;
        InputCount = Curr->OutputCount;
        InputCountB = Curr->OutputCount + 1;
    }}


nnfxp_param_stats Nnfxp_GetParamStats(const nnfxp *NN)
{
    nnfxp_param_stats Stats = { 
        .BiasMax = NNFXP_TYPE_MIN,
        .BiasMin = NNFXP_TYPE_MAX,
        .WeightMax = NNFXP_TYPE_MIN,
        .WeightMin = NNFXP_TYPE_MAX,
    };
    for (int i = 0; i < NN->LayerCount; i++)
    {
        nnfxp_layer *Layer = NN->Layers + i;
        for (int k = 0; k < Layer->OutputCount; k++)
        {
            int Col = k*Layer->InputCountB;
            for (int j = 0; j < Layer->InputCount; j++)
            {
                int Index = Col + j;
                Stats.WeightMax = NNFXP__MAX(Stats.WeightMax, Layer->Weights[Index]);
                Stats.WeightMin = NNFXP__MIN(Stats.WeightMin, Layer->Weights[Index]);
            }

            Stats.BiasMax = NNFXP__MAX(Stats.BiasMax, Layer->Weights[Col + Layer->InputCountB - 1]);
            Stats.BiasMin = NNFXP__MIN(Stats.BiasMin, Layer->Weights[Col + Layer->InputCountB - 1]);
        }
    }
    return Stats;
}

/* MSE + L2 regularization */
nnfxp_type Nnfxp_CalcLoss(nnfxp *NN, const nnfxp_type *ExpectedOutputs, int OutputCount, nnfxp_type L2Lambda)
{
    nnfxp_type Sum = 0;
    const nnfxp_type *Outputs = Nnfxp_GetOutputs(NN);
    for (int i = 0; i < OutputCount; i++)
    {
        nnfxp_type Tmp = (ExpectedOutputs[i] - Outputs[i]);
        Sum += NNFXP_MUL(Tmp, Tmp);
    }

    nnfxp_type L2 = 0;
    for (int i = 0; i < NN->LayerCount; i++)
    {
        nnfxp_type Sum = 0;
        nnfxp_layer *Layer = NN->Layers + i;
        for (int h = 0; h < Layer->OutputCount; h++)
        {
            nnfxp_type *WeightPtr = Layer->Weights + h*Layer->InputCountB;
            for (int k = 0; k < Layer->InputCount; k++)
            {
                nnfxp_type Weight = *WeightPtr++;
                Sum += NNFXP_MUL(Weight, Weight);
            }
        }
        L2 += Sum;
    }
    return NNFXP_MUL(Sum, NNFXP(0.5)) + NNFXP_MUL(NNFXP(0.5), NNFXP_MUL(L2, L2Lambda));
}

nnfxp_type *Nnfxp_GetOutputs(nnfxp *NN)
{
    return NN->Layers[NN->LayerCount - 1].Outputs;
}

void Nnfxp_Print(nnfxp *NN)
{
    printf("Inputs: [");
    for (int i = 0; i < NN->InputCount; i++)
    {
        printf("%g ", NNFXP_FLT(NN->Inputs[i]));
    }
    printf("]\n");

    printf("Layers: %d\n", NN->LayerCount);
    for (int i = 0; i < NN->LayerCount; i++)
    {
        const nnfxp_layer *Layer = NN->Layers + i;
        printf("    layer %d: in/out: %d/%d\n", i, Layer->InputCount, Layer->OutputCount);

        printf("        node vals:  [ ");
        for (int k = 0; k < Layer->OutputCount + 1; k++)
            printf("%6.3f ", NNFXP_FLT(Layer->Outputs[k]));
        printf("]\n");

        printf("        node delta: [ ");
        for (int k = 0; k < Layer->OutputCount; k++)
            printf("%6.3f ", NNFXP_FLT(Layer->Deltas[k]));
        printf("]\n");

        printf("        weights:\n");
        for (int k = 0; k < Layer->OutputCount; k++)
        {
            printf("            [ ");
            for (int j = 0; j < Layer->InputCountB; j++)
            {
                printf("%6.3f ", NNFXP_FLT(Layer->Weights[j + k*Layer->InputCountB]));
            }
            printf("]\n");
        }
    }

    printf("Neural network output: [ ");
    for (int i = 0; i < NN->Layers[NN->LayerCount - 1].OutputCount; i++)
        printf("%g ", NNFXP_FLT(NN->Layers[NN->LayerCount - 1].Outputs[i]));
    printf("]\n");

}

#endif
