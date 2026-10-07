#ifndef NNFXP_H
#define NNFXP_H

/* this can be used independently from neuralnet.h */

#include <stdbool.h>
#include <stdint.h>

#ifndef nnfxp_xtype
#  define nnfxp_xtype int32_t
#endif /* nnfxp_xtype */

#ifndef nnfxp_type
#  define nnfxp_type int16_t
#endif /* nnfxp_type */

#ifndef nnfxp_qtype
#  define nnfxp_qtype int8_t
#endif /* nnfxp_qtype */

#ifndef NNFXP_FRACTION_BITS
#  define NNFXP_FRACTION_BITS 8
#endif /* NNFXP_FRACTION_BITS */




typedef struct nnfxp nnfxp;
typedef struct nnfxp_layer nnfxp_layer;
typedef struct nnfxp_feedforward_config nnfxp_feedforward_config;
typedef struct nnfxp_allocator_param nnfxp_allocator_param;
typedef struct nnfxp_config nnfxp_config;
typedef struct nnfxp_param_stats nnfxp_param_stats;
typedef struct nnfxp_model_config nnfxp_model_config;
typedef enum
{
    NNFXP_ALLOCATE,
    NNFXP_FREE,
} nnfxp_allocator_mode;
typedef enum
{
    NNFXP_CONFIG_NONE = 0,
} nnfxp_config_flags;
typedef enum
{
    NNFXP_MODEL_FLAG_NONE = 0,
    NNFXP_MODEL_FLAG_DONT_COPY_WEIGHTS = 1 << 0, /* provided weights will be used by nnfxp without copying */
} nnfxp_model_flags;
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
    nnfxp_config_flags Flags;
    int InputCount;
    int LayerCount;
    const int *NodeCountPerLayer;

    void *AllocatorData;
    nnfxp_allocator_callback AllocatorCallback;
    void *RandData;
    nnfxp_rand_callback RandCallback;
    void *ActivationData;
    nnfxp_activation_callback ActivationCallback;
};

struct nnfxp_model_config
{
    const nnfxp_qtype *Weights;
    nnfxp_model_flags Flags;
    int QScalarInvShift;        /* b in 2^b, use 0 if weights are not quantized */
    nnfxp_type QScalarInv;      /* 1/(quantization scalar) * 2^b, use NNFXP_ONE if weights are not quantized */
};

struct nnfxp_feedforward_config
{
    int InputCount;
    const nnfxp_type *Inputs;
};


void Nnfxp_Create(nnfxp *NN, const nnfxp_config *Config);
void Nnfxp_CreateFromModel(nnfxp *NN, const nnfxp_config *Config, const nnfxp_model_config *ModelConfig);
void Nnfxp_Destroy(nnfxp *NN);

void Nnfxp_Randomize(nnfxp *NN);
void Nnfxp_FeedForward(nnfxp *NN, const nnfxp_feedforward_config *Config);
nnfxp_type Nnfxp_CalcLoss(nnfxp *NN, const nnfxp_type *ExpectedOutputs, int OutputCount, nnfxp_type L2Lambda);

nnfxp_type *Nnfxp_GetOutputs(nnfxp *NN);
void Nnfxp_Print(nnfxp *NN);


#define NNFXP_TYPE_MAX (nnfxp_type)((1llu << (sizeof(nnfxp_type)*8)) - 1)
#define NNFXP_TYPE_MIN (nnfxp_type)(1llu << (sizeof(nnfxp_type)*8))
#define NNFXP_FLT_EXTRACT_FRAC(val) ((val) - (float)(nnfxp_type)(val))

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

#define NNFXP_MUL(a, b) (((nnfxp_xtype)(a) * (b)) >> (NNFXP_FRACTION_BITS))
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
    nnfxp_qtype *Weights;

    /* OutputCount */
    nnfxp_xtype *OutputsX;
    nnfxp_type *OutputActivated;
};

struct nnfxp
{
    int QScalarInvShift;
    nnfxp_type QScalarInv;
    bool ReadOnlyWeights;
    int InputCount;
    int InputCountB;
    int LayerCount;
    nnfxp_type *Inputs;
    nnfxp_layer *Layers;

    int32_t WeightCount;
    nnfxp_qtype *Weights;
    void *Arena;

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

static nnfxp_xtype Nnfxp__DotProductq(const nnfxp_qtype *A, const nnfxp_type *B, int Length, nnfxp_type QScalar, int Shift)
{
    nnfxp_xtype Result = 0;
    for (int i = 0; i < Length; i++)
    {
        Result += ((nnfxp_xtype)A[i] * B[i] * QScalar) >> (Shift);
    }
    return Result;
}


/* NOTE: Out = A*B^T */
static void Nnfxp__MatMulABTq(
    nnfxp_xtype *Out, const nnfxp_qtype *A, const nnfxp_type *BT, int RowA, int ColA, int RowBT, 
    nnfxp_type QScalar, int Shift
) {
    for (int Ca = 0; Ca < ColA; Ca++)
    {
        for (int Rbt = 0; Rbt < RowBT; Rbt++)
        {
            const nnfxp_qtype *RowMatA = A + Ca*RowA;
            const nnfxp_type *ColMatB = BT + Rbt*RowA;
            nnfxp_type Dp = Nnfxp__DotProductq(RowMatA, ColMatB, RowA, QScalar, Shift);
            Out[Ca*RowBT + Rbt] = Dp;
        }
    }
}

static void Nnfxp__Init(
    nnfxp *NN, 
    nnfxp_config_flags Flags,
    int LayerCount, int InputCount, 
    const int32_t *NodeCountPerLayer,
    void *AllocatorData, nnfxp_allocator_callback AllocatorCallback,
    void *RandData, nnfxp_rand_callback RandCallback,
    void *ActivationData, nnfxp_activation_callback ActivationCallback,
    nnfxp_qtype *Weights
) {
    (void)Flags;
    assert(LayerCount >= 1 && "must have at leaast 1 layer (output layer)");
    *NN = (nnfxp) { 
        .InputCount = InputCount,
        .InputCountB = InputCount + 1,
        .LayerCount = LayerCount,
        .ReadOnlyWeights = Weights != NULL,
    };
    if (AllocatorCallback != NULL)
    {
        NN->AllocatorCallback = AllocatorCallback;
        NN->AllocatorData = AllocatorData;
    }
    else
    {
        NN->AllocatorCallback = Nnfxp__DefaultAllocatorCallback;
        NN->AllocatorData = NULL;
    }
    if (RandCallback != NULL)
    {
        NN->RandCallback = RandCallback;
        NN->RandData = RandData;
    }
    else
    {
        NN->RandCallback = Nnfxp__DefaultRandCallback;
        NN->RandData = NULL;
    }
    if (ActivationCallback != NULL)
    {
        NN->ActivationCallback = ActivationCallback;
        NN->ActivationData = ActivationData;
    }
    else
    {
        NN->ActivationCallback = Nnfxp__Sigmoid;
        NN->ActivationData = NULL;
    }

#define NNFXP__ARENA_ALLOC(arena, sizebytes) (void *)((arena += sizebytes), arena - (sizebytes))
    /* allocate needed mem */
    uint8_t *Arena = NULL;
    nnfxp_qtype *WeightArena = NULL;
    {
        intptr_t WeightSizeBytes = 0;
        intptr_t SizeBytes = sizeof(NN->Inputs[0]) * NN->InputCountB
                        + sizeof(NN->Layers[0]) * NN->LayerCount;
        int InputCountB = NN->InputCountB;
        for (int i = 0; i < LayerCount; i++)
        {
            int OutputCount = NodeCountPerLayer[i];
            int WeightCount = OutputCount*InputCountB;

            if (!Weights)
            {
                WeightSizeBytes += WeightCount*sizeof(NN->Layers[0].Weights[0]);
            }
            SizeBytes += WeightCount*sizeof(NN->Layers[0].Weights[0]);
            SizeBytes += (OutputCount + 1)*sizeof(NN->Layers[0].OutputActivated[0]);
            SizeBytes += (OutputCount + 1)*sizeof(NN->Layers[0].OutputsX[0]);

            InputCountB = OutputCount + 1;
        }

        Arena = NNFXP__ALLOC(NN, SizeBytes);
        NN->Arena = Arena;
        NN->Weights = Weights;
        if (!Weights)
        {
            WeightArena = NNFXP__ARENA_ALLOC(Arena, WeightSizeBytes);
            NN->Weights = WeightArena;
        }
    }

    /* allocate needed mem */
    {
        NN->Inputs = NNFXP__ARENA_ALLOC(Arena, sizeof(NN->Inputs[0]) * NN->InputCountB);
        NN->Layers = NNFXP__ARENA_ALLOC(Arena, sizeof(NN->Layers[0]) * NN->LayerCount);
        NN->WeightCount = 0;
        int InputCount = NN->InputCount;
        int InputCountB = NN->InputCountB;
        for (int i = 0; i < LayerCount; i++)
        {
            int OutputCount = NodeCountPerLayer[i];
            int WeightCount = OutputCount*InputCountB;

            NN->WeightCount += WeightCount;
            if (!Weights)
            {
                NN->Layers[i].Weights = NNFXP__ARENA_ALLOC(WeightArena, WeightCount);
            }
            else
            {
                NN->Layers[i].Weights = NNFXP__ARENA_ALLOC(Weights, WeightCount);
            }
            NN->Layers[i].OutputActivated = NNFXP__ARENA_ALLOC(Arena, (OutputCount + 1)*sizeof(NN->Layers[0].OutputActivated[0]));
            NN->Layers[i].OutputActivated[OutputCount] = NNFXP_ONE;
            NN->Layers[i].OutputsX = NNFXP__ARENA_ALLOC(Arena, (OutputCount + 1)*sizeof(NN->Layers[0].OutputsX[0]));
            NN->Layers[i].InputCount = InputCount;
            NN->Layers[i].InputCountB = InputCountB;
            NN->Layers[i].OutputCount = OutputCount;

            InputCount = OutputCount;
            InputCountB = OutputCount + 1;
        }
    }
#undef NN__ARENA_ALLOC
}



void Nnfxp_Create(nnfxp *NN, const nnfxp_config *Config)
{
    Nnfxp__Init(NN, 
        Config->Flags,
        Config->LayerCount, Config->InputCount,
        Config->NodeCountPerLayer,
        Config->AllocatorData, Config->AllocatorCallback, 
        Config->RandData, Config->RandCallback,
        Config->ActivationData, Config->ActivationCallback,
        NULL
    );
    NN->QScalarInv = NNFXP_ONE;
    Nnfxp_Randomize(NN);
}

void Nnfxp_CreateFromModel(nnfxp *NN, const nnfxp_config *Config, const nnfxp_model_config *ModelConfig)
{
    assert(ModelConfig->Weights);
    if (ModelConfig->Flags & NNFXP_MODEL_FLAG_DONT_COPY_WEIGHTS)
    {
        /* init with weights treated as weak pointer (no copying and no ownership) */
        Nnfxp__Init(NN, 
            Config->Flags,
            Config->LayerCount, Config->InputCount,
            Config->NodeCountPerLayer,
            Config->AllocatorData, Config->AllocatorCallback, 
            Config->RandData, Config->RandCallback,
            Config->ActivationData, Config->ActivationCallback,
            (nnfxp_qtype *)ModelConfig->Weights
        );
    }
    else
    {
        /* init with weights copied */
        Nnfxp__Init(NN, 
            Config->Flags,
            Config->LayerCount, Config->InputCount,
            Config->NodeCountPerLayer,
            Config->AllocatorData, Config->AllocatorCallback, 
            Config->RandData, Config->RandCallback,
            Config->ActivationData, Config->ActivationCallback,
            NULL
        );
        memcpy(NN->Weights, ModelConfig->Weights, NN->WeightCount*sizeof(NN->Weights[0]));
    }
    NN->QScalarInv = ModelConfig->QScalarInv;
    NN->QScalarInvShift = ModelConfig->QScalarInvShift;
}

void Nnfxp_Destroy(nnfxp *NN)
{
    NNFXP__FREE(NN, NN->Arena);
}

void Nnfxp_Randomize(nnfxp *NN)
{
    NN->Inputs[NN->InputCountB - 1] = NNFXP_ONE;
    for (int i = 0; i < NN->LayerCount; i++)
    {
        nnfxp_layer *Layer = NN->Layers + i;
        for (int k = 0; k < Layer->OutputCount; k++)
        {
            Layer->OutputActivated[k] = NNFXP__RAND(NN);
            for (int j = 0; j < Layer->InputCountB; j++)
            {
                Layer->Weights[k*Layer->InputCountB + j] = NNFXP__RAND(NN);
            }
        }
        /* NOTE: for biases */
        Layer->OutputActivated[Layer->OutputCount] = NNFXP_ONE;
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

        Nnfxp__MatMulABTq(
            Layer->OutputsX, Layer->Weights, X,
            Layer->InputCountB, Layer->OutputCount, 1,
            NN->QScalarInv, NN->QScalarInvShift
        );

        /* NOTE: normalize outputs via activation fn ("squish" Y from -inf..+inf to 0..1) */
        for (int r = 0; r < Layer->OutputCount; r++)
        {
            Layer->OutputActivated[r] = ActivationFn(NN->ActivationData, Layer->OutputsX[r]);
        }

        X = Layer->OutputActivated;
    }
}


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

    nnfxp_xtype L2 = 0;
    for (int i = 0; i < NN->LayerCount; i++)
    {
        nnfxp_xtype Sum = 0;
        nnfxp_layer *Layer = NN->Layers + i;
        for (int h = 0; h < Layer->OutputCount; h++)
        {
            nnfxp_qtype *WeightPtr = Layer->Weights + h*Layer->InputCountB;
            for (int k = 0; k < Layer->InputCount; k++)
            {
                nnfxp_type Weight = *WeightPtr++;
                Weight = (Weight * NN->QScalarInv) >> NN->QScalarInvShift;
                Sum += NNFXP_MUL(Weight, Weight);
            }
        }
        L2 += Sum;
    }
    return NNFXP_MUL(Sum, NNFXP(0.5)) + NNFXP_MUL(NNFXP(0.5), NNFXP_MUL(L2, L2Lambda));
}

nnfxp_type *Nnfxp_GetOutputs(nnfxp *NN)
{
    return NN->Layers[NN->LayerCount - 1].OutputActivated;
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
            printf("%6.3f ", NNFXP_FLT(Layer->OutputActivated[k]));
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
        printf("%g ", NNFXP_FLT(NN->Layers[NN->LayerCount - 1].OutputActivated[i]));
    printf("]\n");

}

#endif
