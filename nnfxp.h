#ifndef NNFXP_H
#define NNFXP_H

/* this can be used independently from neuralnet.h */
/* TODO: memory alignment */

#include <stdbool.h>
#include <stdint.h>


/* common type used for stats and not the neural network itself */
#ifndef nnfxp_type
#  define nnfxp_type int32_t
#endif /* nnfxp_type */



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
typedef enum
{
    NNFXP_TYPE_32x32q = 0,
    NNFXP_TYPE_32x16q = 1,
    NNFXP_TYPE_32x8q = 2,
    NNFXP_TYPE_16x16q = 3,
    NNFXP_TYPE_16x8q = 4,
    NNFXP_TYPE_8x8q = 5,
} nnfxp_type_combo;
typedef void *(*nnfxp_allocator_callback)(void *UserData, nnfxp_allocator_param *Param);
typedef void (*nnfxp_activation_callback)(void *UserData, void *Vector, int ElemCount, int ElemSize);

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
    int Alignment; /* alignment of rows in weights and input/output vector */
    const int *NodeCountPerLayer;

    void *AllocatorData;
    nnfxp_allocator_callback AllocatorCallback;
    void *ActivationData;
    nnfxp_activation_callback ActivationCallback;

    int FxpDecimal;
    nnfxp_type_combo TypeCombo;
};

struct nnfxp_model_config
{
    nnfxp_model_flags Flags;
    int WeightCount;
    const void *Weights;
    int RowAlignment;           /* alignment of each row in the weight matrix, provide 0 if it is the same as nnfxp_config.Alignment */
    int QScalarInvShift;        /* b in 2^b, use 0 if weights are not quantized */
    nnfxp_type QScalarInv;      /* 1/(quantization scalar) * 2^b, defaults to 1 (NNFXP_ONE) if <= 0 was given */
};

struct nnfxp_feedforward_config
{
    int InputCount;
    const void *Inputs;
};


/* returns true if overflow occurred, false otherwise */
bool Nnfxp_CreateFromFp32Model(nnfxp *NN, const nnfxp_config *Config, const nnfxp_model_config *ModelConfig);
void Nnfxp_CreateFromIntModel(nnfxp *NN, const nnfxp_config *Config, const nnfxp_model_config *ModelConfig);
void Nnfxp_Destroy(nnfxp *NN);

void Nnfxp_FeedForward(nnfxp *NN, const nnfxp_feedforward_config *Config);
nnfxp_type Nnfxp_CalcLoss(nnfxp *NN, const nnfxp_type *ExpectedOutputs, int OutputCount, nnfxp_type L2Lambda);

void *Nnfxp_GetOutputs(nnfxp *NN);
nnfxp_type Nnfxp_GetOutput(const nnfxp *NN, intptr_t Index);
void Nnfxp_Print(nnfxp *NN);

intptr_t Nnfxp_AlignUp(intptr_t Value, intptr_t Alignment);


#define NNFXP_TYPE_MAX (nnfxp_type)((1llu << (sizeof(nnfxp_type)*8)) - 1)
#define NNFXP_TYPE_MIN (nnfxp_type)(1llu << (sizeof(nnfxp_type)*8))
#define NNFXP_FLT_EXTRACT_FRAC(val) ((val) - (float)(nnfxp_type)(val))

#define NNFXP_EXTRACT_FRAC(fxp) ((fxp) & (NNFXP_ONE - 1))
#define NNFXP_EXTRACT_INT(fxp, decimal) ((fxp) >> (decimal))

#define NNFXP_ONE(decimal) ((nnfxp_type)1 << (decimal))
#define NNFXP(constant, decimal) (nnfxp_type)((constant) * NNFXP_ONE(decimal))
#define NNFXP_FLT(fxp, decimal) ((float)(fxp) / NNFXP_ONE(decimal))

#define NNFXP_ADD(a, b) ((a) + (b))
#define NNFXP_SUB(a, b) ((a) - (b))
#define NNFXP_ADDI(x, i, decimal) NNFXP_ADD(x, (i)*NNFXP_ONE(decimal))
#define NNFXP_SUBI(x, i, decimal) NNFXP_SUB(x, (i)*NNFXP_ONE(decimal))
#define NNFXP_ADDC(x, constant, decimal) NNFXP_ADD(x, NNFXP(constant, decimal), decimal) 
#define NNFXP_SUBC(x, constant, decimal) NNFXP_SUB(x, NNFXP(constant, decimal), decimal)

#define NNFXP_MUL(a, b, decimal) (((nnfxp_type)(a) * (b)) >> (decimal))
#define NNFXP_DIV(a, b, decimal) (((a) / (b)) << decimal)
#define NNFXP_DIVL(a, b, decimal) ((((a) << (decimal)) / (b)))
#define NNFXP_DIVR(a, b, decimal) (((a) / ((b) >> (decimal))))
#define NNFXP_MULI(x, i) ((x) * (i))
#define NNFXP_DIVI(x, i) ((x) / (i))




struct nnfxp_layer
{
    int InputCount;
    int InputCountB;
    int OutputCount;
    /* InputCountB x OutputCount */
    void *Weights;

    /* OutputCount */
    void *Outputs;
};

struct nnfxp
{
    void *Arena;

    int Alignment;
    int FxpDecimal;
    nnfxp_type_combo TypeCombo;
    int QScalarInvShift;
    nnfxp_type QScalarInv;
    bool ReadOnlyWeights;

    int InputCount;
    int InputCountB;
    int LayerCount;
    void *Inputs;
    nnfxp_layer *Layers;

    int WeightCount;
    int WeightElemSize;
    void *Weights;

    void *AllocatorData;
    nnfxp_allocator_callback AllocatorCallback;
    void *ActivationData;
    nnfxp_activation_callback ActivationCallback;

    void *MatMul;
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


#define NNFXP__DEFINE(dp_name, matmul_name, dequantized_type, quantized_type) \
static dequantized_type dp_name(\
    const quantized_type *A, const dequantized_type *B, int Length, nnfxp_type QScalar, int Shift\
) {\
    dequantized_type Result = 0;\
    for (int i = 0; i < Length; i++) {\
        Result += ((dequantized_type)A[i] * B[i] * QScalar) >> (Shift);\
    }\
    return Result;\
} \
\
static void matmul_name(\
    dequantized_type *Out, const quantized_type *A, const dequantized_type *BT, int RowA, int ColA, int RowBT, \
    nnfxp_type QScalar, int Shift, int StrideA, int StrideBT\
) {\
    if (!StrideA) StrideA = RowA;\
    if (!StrideBT) StrideBT = RowBT;\
    for (int Ca = 0; Ca < ColA; Ca++) {\
        for (int Rbt = 0; Rbt < RowBT; Rbt++) {\
            const quantized_type *RowMatA = A + Ca*StrideA;\
            const dequantized_type *ColMatB = BT + Rbt*StrideBT;\
            dequantized_type Dp = dp_name(RowMatA, ColMatB, RowA, QScalar, Shift);\
            Out[Ca*StrideBT + Rbt] = Dp;\
        }\
    }\
}\
static void matmul_name(\
    dequantized_type *Out, const quantized_type *A, const dequantized_type *BT, int RowA, int ColA, int RowBT, \
    nnfxp_type QScalar, int Shift, int Stride, int StrideBT\
)

NNFXP__DEFINE(Nnfxp__Dp_32x32q, Nnfxp__MatMulABT_32x32q, int32_t, int32_t);
NNFXP__DEFINE(Nnfxp__Dp_32x16q, Nnfxp__MatMulABT_32x16q, int32_t, int16_t);
NNFXP__DEFINE(Nnfxp__Dp_32x8q, Nnfxp__MatMulABT_32x8q, int32_t, int8_t);
NNFXP__DEFINE(Nnfxp__Dp_16x16q, Nnfxp__MatMulABT_16x16q, int16_t, int16_t);
NNFXP__DEFINE(Nnfxp__Dp_16x8q, Nnfxp__MatMulABT_16x8q, int16_t, int8_t);
NNFXP__DEFINE(Nnfxp__Dp_8x8q, Nnfxp__MatMulABT_8x8q, int8_t, int8_t);

#undef NNFXP__DEFINE

#define NNFXP__DQ_SIZE(p_nn) (Nnfxp__GetDqTypeSize((p_nn)->TypeCombo))
#define NNFXP__Q_SIZE(p_nn) (Nnfxp__GetQTypeSize((p_nn)->TypeCombo))
#define NNFXP__ONE(p_nn) NNFXP_ONE((p_nn)->FxpDecimal)
#define NNFXP__ALIGN_UP(value) Nnfxp_AlignUp(value, NN->Alignment)
#define NNFXP__ALIGN_PTR_UP(ptr) (typeof(ptr))Nnfxp_AlignUp((intptr_t)(ptr), NN->Alignment)



typedef void (*nnfxp__matmul_callback)(
    void *Dst, const void *A, const void *BT, 
    int RowA, int ColA, int RowBT, nnfxp_type QScalar, int Shift,
    int StrideA, int StrideB
);


intptr_t Nnfxp_AlignUp(intptr_t Value, intptr_t Alignment)
{
    if ((Value & (Alignment - 1)) == 0) 
        return Value;
    intptr_t Result = (Value + Alignment) - (Value & (Alignment - 1));
    return Result;
}

static int Nnfxp__GetDqTypeSize(nnfxp_type_combo Combo)
{
    switch (Combo)
    {
    case NNFXP_TYPE_32x32q:
    case NNFXP_TYPE_32x16q: 
    case NNFXP_TYPE_32x8q:
        return sizeof(int32_t);
    case NNFXP_TYPE_16x16q:
    case NNFXP_TYPE_16x8q: 
        return sizeof(int16_t);
    case NNFXP_TYPE_8x8q:
        return sizeof(int8_t);
    }
    return 0;
}

static int Nnfxp__GetQTypeSize(nnfxp_type_combo Combo)
{
    switch (Combo)
    {
    case NNFXP_TYPE_32x32q:
        return sizeof(int32_t);
    case NNFXP_TYPE_32x16q: 
    case NNFXP_TYPE_16x16q:
        return sizeof(int16_t);
    case NNFXP_TYPE_32x8q:
    case NNFXP_TYPE_16x8q: 
    case NNFXP_TYPE_8x8q:
        return sizeof(int8_t);
    }
    return 0;
}


static nnfxp_type Nnfxp__Sext(nnfxp_type Value, int SizeBytes)
{
    switch (SizeBytes)
    {
    case 1: return (int8_t)Value;
    case 2: return (int16_t)Value;
    default: return Value;
    }
}

static void Nnfxp__DqStore(const nnfxp *NN, void *Dst, intptr_t Index, int32_t Value)
{
    int ElemSize = NNFXP__DQ_SIZE(NN);
    uint8_t *Ptr = Dst;
    Ptr += Index*ElemSize;
    /* NOTE: assuming LE */
    memcpy(Ptr, &Value, ElemSize);
}

static void Nnfxp__QStore(const nnfxp *NN, void *Dst, intptr_t Index, int32_t Value)
{
    int ElemSize = NNFXP__Q_SIZE(NN);
    uint8_t *Ptr = Dst;
    Ptr += Index*ElemSize;
    memcpy(Ptr, &Value, ElemSize);
}

/* load a quantized value, returns the dequantized version */
static nnfxp_type Nnfxp__QLoadDQ(const nnfxp *NN, const void *Src, intptr_t Index)
{
    int ElemSize = NNFXP__Q_SIZE(NN);
    const uint8_t *Ptr = Src;
    Ptr += Index*ElemSize;

    nnfxp_type Result = 0;
    memcpy(&Result, Ptr, ElemSize);
    Result = Nnfxp__Sext(Result, ElemSize);
    Result = (Result * NN->QScalarInv) >> NN->QScalarInvShift;

    return Result;
}

static nnfxp_type Nnfxp__DqLoad(const nnfxp *NN, const void *Src, intptr_t Index)
{
    int ElemSize = NNFXP__DQ_SIZE(NN);
    const uint8_t *Ptr = Src;
    Ptr += Index*ElemSize;

    nnfxp_type Result = 0;
    memcpy(&Result, Ptr, ElemSize);
    Result = Nnfxp__Sext(Result, ElemSize);
    return Result;
}

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

static void Nnfxp__Sigmoid(void *Data, void *Vector, int VectorLength, int ElemSize)
{
    (void)ElemSize;
    const nnfxp *NN = Data;
    for (int i = 0; i < VectorLength; i++)
    {
        nnfxp_type X = Nnfxp__DqLoad(NN, Vector, i);
        nnfxp_type Y = 0;
        /* piecewise approx of 1/(1 + e^-x) */
        if (NNFXP__IN_RANGE(NNFXP(-1.0, NN->FxpDecimal), X, NNFXP(1.0, NN->FxpDecimal)))
        {
            Y = NNFXP(0.5, NN->FxpDecimal) + NNFXP_MUL(NNFXP(0.235, NN->FxpDecimal), X, NN->FxpDecimal);
        }
        else if (NNFXP__IN_RANGE(NNFXP(-1.5, NN->FxpDecimal), X, NNFXP(-1.0, NN->FxpDecimal)))
        {
            Y = NNFXP(0.445, NN->FxpDecimal) + NNFXP_MUL(NNFXP(0.175, NN->FxpDecimal), X, NN->FxpDecimal);
        }
        else if (NNFXP__IN_RANGE(NNFXP(1.0, NN->FxpDecimal), X, NNFXP(1.5, NN->FxpDecimal)))
        {
            Y = NNFXP(0.555, NN->FxpDecimal) + NNFXP_MUL(NNFXP(0.175, NN->FxpDecimal), X, NN->FxpDecimal);
        }
        else if (NNFXP__IN_RANGE(NNFXP(-2.0, NN->FxpDecimal), X, NNFXP(-1.5, NN->FxpDecimal)))
        {
            Y = NNFXP(0.3708, NN->FxpDecimal) + NNFXP_MUL(NNFXP(0.126, NN->FxpDecimal), X, NN->FxpDecimal);
        }
        else if (NNFXP__IN_RANGE(NNFXP(1.5, NN->FxpDecimal), X, NNFXP(2.0, NN->FxpDecimal)))
        {
            Y = NNFXP(0.6308, NN->FxpDecimal) + NNFXP_MUL(NNFXP(0.126, NN->FxpDecimal), X, NN->FxpDecimal);
        }
        else if (NNFXP__IN_RANGE(NNFXP(-3.0, NN->FxpDecimal), X, NNFXP(-2.0, NN->FxpDecimal)))
        {
            Y = NNFXP(0.26, NN->FxpDecimal) + NNFXP_MUL(NNFXP(0.07178, NN->FxpDecimal), X, NN->FxpDecimal);
        }
        else if (NNFXP__IN_RANGE(NNFXP(2.0, NN->FxpDecimal), X, NNFXP(3.0, NN->FxpDecimal)))
        {
            Y = NNFXP(0.74, NN->FxpDecimal) + NNFXP_MUL(NNFXP(0.07178, NN->FxpDecimal), X, NN->FxpDecimal);
        }
        else if (NNFXP__IN_RANGE(NNFXP(-5.0, NN->FxpDecimal), X, NNFXP(-3.0, NN->FxpDecimal)))
        {
            Y = NNFXP(0.105, NN->FxpDecimal) + NNFXP_MUL(NNFXP(0.0203665, NN->FxpDecimal), X, NN->FxpDecimal);
        }
        else if (NNFXP__IN_RANGE(NNFXP(3.0, NN->FxpDecimal), X, NNFXP(5.0, NN->FxpDecimal)))
        {
            Y = NNFXP(0.895, NN->FxpDecimal) + NNFXP_MUL(NNFXP(0.0203665, NN->FxpDecimal), X, NN->FxpDecimal);
        }
        else if (X > NNFXP(5, NN->FxpDecimal))
        {
            Y = NNFXP__ONE(NN);
        }
        Nnfxp__DqStore(NN, Vector, i, Y);
    }
}

static int Nnfxp__GetWeightStride(const nnfxp *NN, int InputCountB)
{
    int QSize = NNFXP__Q_SIZE(NN);
    return NNFXP__ALIGN_UP(InputCountB * QSize) / QSize;
}

static void Nnfxp__Init(
    nnfxp *NN, 
    const nnfxp_config *Config,
    const void *Weights,
    nnfxp_type QScalarInv, int QScalarInvShift
) {
    assert(Config->LayerCount >= 1 && "must have at leaast 1 layer (output layer)");
    assert((Config->Alignment & (Config->Alignment - 1)) == 0 && "Alignment must be a power of 2");
    *NN = (nnfxp) { 
        .InputCount = Config->InputCount,
        .InputCountB = Config->InputCount + 1,
        .LayerCount = Config->LayerCount,
        .ReadOnlyWeights = Weights != NULL,

        .QScalarInv = QScalarInv <= 0? NNFXP_ONE(Config->FxpDecimal) : QScalarInv,
        .QScalarInvShift = QScalarInvShift,
        .TypeCombo = Config->TypeCombo,
        .FxpDecimal = Config->FxpDecimal,
        .WeightElemSize = Nnfxp__GetQTypeSize(Config->TypeCombo),
        .Alignment = Config->Alignment > 0? Config->Alignment : 1,
    };
    switch (Config->TypeCombo)
    {
    case NNFXP_TYPE_32x32q: NN->MatMul = Nnfxp__MatMulABT_32x32q; break;
    case NNFXP_TYPE_32x16q: NN->MatMul = Nnfxp__MatMulABT_32x16q; break;
    case NNFXP_TYPE_32x8q: NN->MatMul = Nnfxp__MatMulABT_32x8q; break;
    case NNFXP_TYPE_16x16q: NN->MatMul = Nnfxp__MatMulABT_16x16q; break;
    case NNFXP_TYPE_16x8q: NN->MatMul = Nnfxp__MatMulABT_16x8q; break;
    case NNFXP_TYPE_8x8q: NN->MatMul = Nnfxp__MatMulABT_8x8q; break;
    }

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
    if (Config->ActivationCallback != NULL)
    {
        NN->ActivationCallback = Config->ActivationCallback;
        NN->ActivationData = Config->ActivationData;
    }
    else
    {
        NN->ActivationCallback = Nnfxp__Sigmoid;
        NN->ActivationData = NN;
    }

#define NNFXP__ARENA_ALLOC(arena, sizebytes) (void *)((arena += sizebytes), arena - (sizebytes))
    /* calc needed mem */
    uint8_t *Arena = NULL;
    uint8_t *WeightArena = NULL;
    {
        intptr_t SizeBytes = NNFXP__ALIGN_UP(NNFXP__DQ_SIZE(NN) * NN->InputCountB);
        SizeBytes += NNFXP__ALIGN_UP(sizeof(NN->Layers[0]) * NN->LayerCount);
        int InputCountB = NN->InputCountB;

        intptr_t TotalWeightSizeBytes = 0;
        for (int i = 0; i < Config->LayerCount; i++)
        {
            int OutputCount = Config->NodeCountPerLayer[i];

            if (!Weights)
            {
                int WeightSizeBytes = OutputCount*NNFXP__ALIGN_UP(InputCountB*NNFXP__Q_SIZE(NN));
                TotalWeightSizeBytes += WeightSizeBytes;
                SizeBytes += WeightSizeBytes;
            }
            int OutputSizeBytes = NNFXP__ALIGN_UP((OutputCount + 1)*NNFXP__DQ_SIZE(NN));
            SizeBytes += OutputSizeBytes;

            InputCountB = OutputCount + 1;
        }

        SizeBytes = NNFXP__ALIGN_UP(SizeBytes + NN->Alignment);
        NN->Arena = NNFXP__ALLOC(NN, SizeBytes);
        Arena = NNFXP__ALIGN_PTR_UP(NN->Arena);

        if (!Weights)
        {
            WeightArena = NNFXP__ARENA_ALLOC(Arena, TotalWeightSizeBytes);
        }
        else
        {
            WeightArena = (uint8_t *)Weights; /* NOTE: const cast */
        }
        NN->Weights = WeightArena;
    }

    /* allocate needed mem */
    {
        {
            intptr_t InputSizeBytes = NNFXP__ALIGN_UP(NNFXP__DQ_SIZE(NN) * NN->InputCountB);
            NN->Inputs = NNFXP__ARENA_ALLOC(Arena, InputSizeBytes);
        }

        intptr_t LayerSizeBytes = NNFXP__ALIGN_UP(sizeof(NN->Layers[0]) * NN->LayerCount);
        NN->Layers = NNFXP__ARENA_ALLOC(Arena, LayerSizeBytes);

        NN->WeightCount = 0;
        int InputCount = NN->InputCount;
        int InputCountB = NN->InputCountB;
        for (int i = 0; i < NN->LayerCount; i++)
        {
            int OutputCount = Config->NodeCountPerLayer[i];
            int WeightSizeBytes = OutputCount*NNFXP__ALIGN_UP(InputCountB*NNFXP__Q_SIZE(NN));
            int OutputSizeBytes = NNFXP__ALIGN_UP((OutputCount + 1)*NNFXP__DQ_SIZE(NN));

            NN->WeightCount += OutputCount*InputCountB;
            NN->Layers[i].Weights = NNFXP__ARENA_ALLOC(WeightArena, WeightSizeBytes);
            NN->Layers[i].Outputs = NNFXP__ARENA_ALLOC(Arena, OutputSizeBytes);
            NN->Layers[i].InputCount = InputCount;
            NN->Layers[i].InputCountB = InputCountB;
            NN->Layers[i].OutputCount = OutputCount;

            InputCount = OutputCount;
            InputCountB = OutputCount + 1;
        }
    }
#undef NN__ARENA_ALLOC
}


bool Nnfxp_CreateFromFp32Model(nnfxp *NN, const nnfxp_config *Config, const nnfxp_model_config *ModelConfig)
{
    Nnfxp__Init(
        NN, Config, NULL, 
        ModelConfig->QScalarInv, ModelConfig->QScalarInvShift
    );
    assert(ModelConfig->Weights);
    assert(ModelConfig->WeightCount == NN->WeightCount && "Mismatched weight count");
    int ModelAlignment = ModelConfig->RowAlignment > 0? ModelConfig->RowAlignment : 1;

    int QTypeSize = Nnfxp__GetQTypeSize(NN->TypeCombo);
    float QTypeMax = ((1ll << QTypeSize*8) - 1);
    /* NOTE: scalar quantization needs to be symmetric => QTypeMin = -QTypeMax */

    const float *WeightPtr = ModelConfig->Weights;
    float Scalar = (float)(1ll << ModelConfig->QScalarInvShift) / NN->QScalarInv;
    bool Overflowed = false;
    for (int i = 0; i < NN->LayerCount; i++)
    {
        nnfxp_layer *Layer = NN->Layers + i;
        int SrcStride = Nnfxp_AlignUp(Layer->InputCountB*sizeof(WeightPtr[0]), ModelAlignment) / sizeof(WeightPtr[0]);
        int DstStride = Nnfxp__GetWeightStride(NN, Layer->InputCountB);

        for (int k = 0; k < Layer->OutputCount; k++)
        {
            for (int j = 0; j < Layer->InputCountB; j++)
            {
                int SrcIndex = k*SrcStride + j;
                int DstIndex = k*DstStride + j;

                float QWeight = WeightPtr[SrcIndex] * Scalar;
                if (QWeight > QTypeMax)
                    QWeight = QTypeMax;
                else if (QWeight < -QTypeMax)
                    QWeight = -QTypeMax;
                Overflowed = !NNFXP__IN_RANGE(-QTypeMax, QWeight, QTypeMax);

                nnfxp_type IntQWeight = QWeight;
                Nnfxp__QStore(NN, Layer->Weights, DstIndex, IntQWeight);
            }
        }
        WeightPtr += Layer->OutputCount*SrcStride;
    }
    return Overflowed;
}

void Nnfxp_CreateFromIntModel(nnfxp *NN, const nnfxp_config *Config, const nnfxp_model_config *ModelConfig)
{
    assert(ModelConfig->Weights);
    if (ModelConfig->Flags & NNFXP_MODEL_FLAG_DONT_COPY_WEIGHTS)
    {
        assert((ModelConfig->RowAlignment == 0 || ModelConfig->RowAlignment == Config->Alignment)
            && "Mismatched alignment"
        );
        /* init with weights treated as weak pointer (no copying and no ownership) */
        Nnfxp__Init(
            NN, Config, ModelConfig->Weights, 
            ModelConfig->QScalarInv, ModelConfig->QScalarInvShift
        );
    }
    else
    {
        /* init with weights copied */
        Nnfxp__Init(
            NN, Config, NULL,
            ModelConfig->QScalarInv, ModelConfig->QScalarInvShift
        );
        if (ModelConfig->RowAlignment == Config->Alignment || ModelConfig->RowAlignment == 0)
        {
            memcpy(NN->Weights, ModelConfig->Weights, NN->WeightCount*NNFXP__Q_SIZE(NN));
        }
        else
        {
            assert("TODO: loading int model with different alignment");
        }
    }
    assert(ModelConfig->WeightCount == NN->WeightCount && "Mismatched weight count");
}

void Nnfxp_Destroy(nnfxp *NN)
{
    NNFXP__FREE(NN, NN->Arena);
}


void Nnfxp_FeedForward(nnfxp *NN, const nnfxp_feedforward_config *Config)
{
    assert(Config->Inputs && Config->InputCount == NN->InputCount);
    memcpy(NN->Inputs, Config->Inputs, Config->InputCount*NNFXP__DQ_SIZE(NN));

    nnfxp_activation_callback ActivateVector = NN->ActivationCallback;
    assert(ActivateVector);

    nnfxp__matmul_callback MatMulABT = NN->MatMul;

    void *X = NN->Inputs;
    for (int i = 0; i < NN->LayerCount; i++)
    {
        nnfxp_layer *Layer = &NN->Layers[i];
        int WStride = Nnfxp__GetWeightStride(NN, Layer->InputCountB);
        int XStride = 1;
        Nnfxp__DqStore(NN, X, Layer->InputCountB - 1, NNFXP__ONE(NN));

        MatMulABT(
            Layer->Outputs, Layer->Weights, X,
            Layer->InputCountB, Layer->OutputCount, XStride,
            NN->QScalarInv, NN->QScalarInvShift,
            WStride, XStride
        );
        ActivateVector(
            NN->ActivationData, Layer->Outputs, Layer->OutputCount, NNFXP__DQ_SIZE(NN)
        );

        X = Layer->Outputs;
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
            int Col = k*Nnfxp__GetWeightStride(NN, Layer->InputCountB);
            for (int j = 0; j < Layer->InputCount; j++)
            {
                int Index = Col + j;
                Stats.WeightMax = NNFXP__MAX(Stats.WeightMax, Nnfxp__QLoadDQ(NN, Layer->Weights, Index));
                Stats.WeightMin = NNFXP__MIN(Stats.WeightMin, Nnfxp__QLoadDQ(NN, Layer->Weights, Index));
            }

            Stats.BiasMax = NNFXP__MAX(Stats.BiasMax, Nnfxp__QLoadDQ(NN, Layer->Weights, Col + Layer->InputCountB - 1));
            Stats.BiasMin = NNFXP__MIN(Stats.BiasMin, Nnfxp__QLoadDQ(NN, Layer->Weights, Col + Layer->InputCountB - 1));
        }
    }
    return Stats;
}

/* MSE + L2 regularization */
nnfxp_type Nnfxp_CalcLoss(nnfxp *NN, const nnfxp_type *ExpectedOutputs, int OutputCount, nnfxp_type L2Lambda)
{
    nnfxp_type Sum = 0;
    const void *Outputs = Nnfxp_GetOutputs(NN);
    for (int i = 0; i < OutputCount; i++)
    {
        nnfxp_type Tmp = (ExpectedOutputs[i] - Nnfxp__DqLoad(NN, Outputs, i));
        Sum += NNFXP_MUL(Tmp, Tmp, NN->FxpDecimal);
    }

    nnfxp_type L2 = 0;
    for (int i = 0; i < NN->LayerCount; i++)
    {
        nnfxp_type Sum = 0;
        nnfxp_layer *Layer = NN->Layers + i;
        for (int h = 0; h < Layer->OutputCount; h++)
        {
            for (int k = 0; k < Layer->InputCount; k++)
            {
                int Index = h*Nnfxp__GetWeightStride(NN, Layer->InputCountB) + k;
                nnfxp_type Weight = Nnfxp__QLoadDQ(NN, Layer->Weights, Index);
                Sum += NNFXP_MUL(Weight, Weight, NN->FxpDecimal);
            }
        }
        L2 += Sum;
    }
    return NNFXP_MUL(
            Sum, 
            NNFXP(0.5, NN->FxpDecimal), NN->FxpDecimal) 
        + NNFXP_MUL(
            NNFXP(0.5, NN->FxpDecimal), 
            NNFXP_MUL(L2, L2Lambda, NN->FxpDecimal), 
            NN->FxpDecimal
        );
}

void *Nnfxp_GetOutputs(nnfxp *NN)
{
    return NN->Layers[NN->LayerCount - 1].Outputs;
}

nnfxp_type Nnfxp_GetOutput(const nnfxp *NN, intptr_t Index)
{
    return Nnfxp__DqLoad(NN, Nnfxp_GetOutputs((nnfxp *)NN), Index);
}

void Nnfxp_Print(nnfxp *NN)
{
    printf("Inputs: [");
    for (int i = 0; i < NN->InputCount; i++)
    {
        nnfxp_type Value = Nnfxp__DqLoad(NN, NN->Inputs, i);
        printf("%g ", NNFXP_FLT(Value, NN->FxpDecimal));
    }
    printf("]\n");

    printf("Layers: %d\n", NN->LayerCount);
    for (int i = 0; i < NN->LayerCount; i++)
    {
        const nnfxp_layer *Layer = NN->Layers + i;
        printf("    layer %d: in/out: %d/%d\n", i, Layer->InputCount, Layer->OutputCount);

        printf("        node vals:  [ ");
        for (int k = 0; k < Layer->OutputCount + 1; k++)
        {
            nnfxp_type Value = Nnfxp__DqLoad(NN, Layer->Outputs, k);
            printf("%6.3f ", NNFXP_FLT(Value, NN->FxpDecimal));
        }
        printf("]\n");

        printf("        weights:\n");
        for (int k = 0; k < Layer->OutputCount; k++)
        {
            printf("            [ ");
            for (int j = 0; j < Layer->InputCountB; j++)
            {
                nnfxp_type Value = Nnfxp__QLoadDQ(NN, Layer->Weights, j + k*Nnfxp__GetWeightStride(NN, Layer->InputCountB));
                printf("%6.3f ", NNFXP_FLT(Value, NN->FxpDecimal));
            }
            printf("]\n");
        }
    }

    printf("Neural network output: [ ");
    for (int i = 0; i < NN->Layers[NN->LayerCount - 1].OutputCount; i++)
    {
        nnfxp_type Value = Nnfxp__DqLoad(NN, NN->Layers[NN->LayerCount - 1].Outputs, i);
        printf("%g ", NNFXP_FLT(Value, NN->FxpDecimal));
    }
    printf("]\n");

}

#endif
