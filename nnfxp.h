#ifndef NNFXP_H
#define NNFXP_H

/* this can be used independently from neuralnet.h */

#include <stdbool.h>

#ifndef nnfxp_type
#  include <stdint.h>
#  define NNFXP_TYPE_MAX INT8_MAX
#  define NNFXP_TYPE_MIN INT8_MIN
#  define nnfxp_type int8_t
#endif /* nnfxp_type */

#ifndef NNFXP_FRACTION_BITS
#  define NNFXP_FRACTION_BITS 5
#endif /* NNFXP_FRACTION_BITS */




typedef struct nnfxp nnfxp;
typedef struct nnfxp_layer nnfxp_layer;
typedef struct nnfxp_feedforward_config nnfxp_feedforward_config;
typedef struct nnfxp_allocator_param nnfxp_allocator_param;
typedef struct nnfxp_config nnfxp_config;
typedef struct nnfxp_backprop_config nnfxp_backprop_config;
typedef enum
{
    NNFXP_ALLOCATE,
    NNFXP_FREE,
} nnfxp_allocator_mode;
typedef void *(*nnfxp_allocator_callback)(void *UserData, nnfxp_allocator_param *Param);
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

struct nnfxp_config
{
    int InputCount;
    int LayerCount;
    int *NodeCountPerLayer;

    void *AllocatorData;
    nnfxp_allocator_callback AllocatorCallback;

    void *RandData;
    nnfxp_rand_callback RandCallback;
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
float Nnfxp_CalcLoss(nnfxp *NN, const nnfxp_type *ExpectedOutputs, int OutputCount, nnfxp_type L2Lambda);

nnfxp_type *Nnfxp_GetOutputs(nnfxp *NN);


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
};


#endif /* NNFXP */


#if defined(NNFXP_IMPLEMENTATION) && !defined(NNFXP_ALREADY_IMPLEMENTED)
#define NNFXP_ALREADY_IMPLEMENTED

#include <stdlib.h>
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

#define NNFXP__SIMD_VEC_LEN 8
#define NNFXP__MAX(a, b) ((a) > (b)? (a) : (b))
#define NNFXP__MIN(a, b) ((a) < (b)? (a) : (b))

#define NNFXP__FLT_EXTRACT_FRAC(val) ((val) - (double)(nnfxp_type)(val))

#define NNFXP__ONE ((nnfxp_type)1 << NNFXP_FRACTION_BITS)
#define NNFXP__EXTRACT_FRAC(fxp) ((fxp) & (NNFXP__ONE - 1))
#define NNFXP__EXTRACT_INT(fxp) ((fxp) >> NNFXP_FRACTION_BITS)

#define NNFXP(constant) (\
        ((nnfxp_type)(constant) << NNFXP_FRACTION_BITS) \
        + NNFXP__FLT_EXTRACT_FRAC(constant) * (double)NNFXP__ONE\
    )

#define NNFXP__FLT(fxp) (\
        (double)((fxp) >> NNFXP_FRACTION_BITS) + (double)NNFXP__EXTRACT_FRAC(fxp) / NNFXP__ONE \
    )

#define NNFXP__ADD(a, b) ((a) + (b))
#define NNFXP__SUB(a, b) ((a) - (b))
#define NNFXP__ADDI(x, i) NNFXP__ADD(x, (i)*NNFXP__ONE)
#define NNFXP__SUBI(x, i) NNFXP__SUB(x, (i)*NNFXP__ONE)
#define NNFXP__ADDC(x, constant) NNFXP__ADD(x, NNFXP(constant)) 
#define NNFXP__SUBC(x, constant) NNFXP__SUB(x, NNFXP(constant))

#define NNFXP__MUL(a, b) (((a) * (b)) >> NNFXP_FRACTION_BITS)
#define NNFXP__DIV(a, b) (((a) / (b)) << NNFXP_FRACTION_BITS)
#define NNFXP__DIVL(a, b) ((((a) << NNFXP_FRACTION_BITS) / (b)))
#define NNFXP__DIVR(a, b) (((a) / ((b) >> NNFXP_FRACTION_BITS)))
#define NNFXP__MULI(x, i) ((x) * (i))
#define NNFXP__DIVI(x, i) ((x) / (i))
#define NNFXP__MULC(x, constant) NNFXP__MUL(x, NNFXP(constant))
#define NNFXP__DIVC(x, constant) NNFXP__DIV(x, NNFXP(constant))




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
    return rand();
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
            NN->Layers[i].Deltas = NNFXP__ALLOC(NN, OutputCount*sizeof(NN->Layers[0].Deltas[0]));
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
    NN->Inputs[NN->InputCountB - 1] = NNFXP__ONE;
    for (int i = 0; i < NN->LayerCount; i++)
    {
        nnfxp_layer *Layer = NN->Layers + i;
        for (int k = 0; k < Layer->OutputCount; k++)
        {
            Layer->Outputs[k] = NNFXP__RAND(NN);
            for (int j = 0; j < Layer->InputCount; j++)
            {
                Layer->Weights[k*Layer->InputCountB + j] = NNFXP__RAND(NN);
            }
        }
        /* NOTE: for biases */
        Layer->Outputs[Layer->OutputCount] = NNFXP__ONE;
    }
}

void Nnfxp_FeedForward(nnfxp *NN, const nnfxp_feedforward_config *Config)
{
}

void Nnfxp_Backprop(nnfxp *NN, const nnfxp_backprop_config *Config)
{
}


#if !defined(NNFXP_USE_SIMD)
#else
#endif

#endif
