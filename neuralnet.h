#ifndef NEURALNET_H
#define NEURALNET_H

typedef struct neuralnet neuralnet;
typedef struct neuralnet_layer neuralnet_layer;
typedef struct neuralnet_config neuralnet_config;
typedef float (*neuralnet_activation_fn)(float NodeBias, float NodeValue);

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
    int WeightCount; /* == (this NodeCount)*(previous NodeCount) */
    float *Weights;

    int NodeCount;
    float *NodeBiases;
    float *NodeValues;
};



neuralnet NeuralNet_Create(const neuralnet_config *Config);
void NeuralNet_Destroy(neuralnet *NN);
void NeuralNet_FeedForward(neuralnet *NN, neuralnet_activation_fn Fn);




#endif /* NEURALNET_H */




#if defined(NEURALNET_IMPLEMENTATION) && !defined(NEURALNET_ALREADY_IMPLEMENTED)
#define NEURALNET_ALREADY_IMPLEMENTED


#include <stdlib.h> /* malloc, free */
#include <stdint.h>
#include <assert.h>
#include <time.h>


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
static void *NN__DefaultAllocatorCallback(void *Data, neuralnet_allocator_param *Param);
static void NN__Randomize(neuralnet *NN);
static float NN__GetRandomValue(void);
static float NN__StepFn(float Bias, float Value);


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
        int PrevLayerNodeCount = Config->InputCount;
        for (int i = 0; i < Config->LayerCount; i++)
        {
            int CurrLayerNodeCount = Config->NodeCountPerLayer[i];
            int WeightCount = PrevLayerNodeCount*CurrLayerNodeCount;
            int BiasCount = CurrLayerNodeCount;

            NN.Layers[i].Weights = NN__ALLOC(&NN, WeightCount*sizeof(NN.Layers[0].Weights[0]));
            NN.Layers[i].NodeBiases = NN__ALLOC(&NN, BiasCount*sizeof(NN.Layers[0].NodeBiases[0]));
            NN.Layers[i].NodeValues = NN__ALLOC(&NN, CurrLayerNodeCount*sizeof(NN.Layers[0].NodeValues[0]));

            PrevLayerNodeCount = CurrLayerNodeCount;
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
    }
    NN__FREE(NN, NN->Layers);
    NN__FREE(NN, NN->Inputs);
}

void NeuralNet_FeedForward(neuralnet *NN, neuralnet_activation_fn ActivationFn)
{
#if 0
    /* NOTE: pseudocode: */
    for each layer:
        for each node - k:
            accum = 0;
            for each prev node - i:
                accum += node[i] * weights[i][k]
            node[k] = activation_fn(bias[k], accum)
#endif


    if (ActivationFn == NULL)
    {
        ActivationFn = NN__StepFn;
    }

    float *PrevNodes = NN->Inputs;
    int PrevNodeCount = NN->InputCount;
    for (int n = 0; n < NN->LayerCount; n++)
    {
        for (int k = 0; k < NN->Layers[n].NodeCount; k++)
        {
            float Accum = 0;
            for (int i = 0; i < PrevNodeCount; i++)
            {
                Accum += PrevNodes[i] * NN->Layers[n].Weights[k*PrevNodeCount + i];
            }

            NN->Layers[n].NodeValues[k] = ActivationFn(
                NN->Layers[n].NodeBiases[k], 
                Accum
            );
        }

        PrevNodes = NN->Layers[n].NodeValues;
        PrevNodeCount = NN->Layers[n].NodeCount;
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
    int PrevNodeCount = NN->InputCount;
    for (int n = 0; n < NN->LayerCount; n++)
    {
        for (int k = 0; k < NN->Layers[n].NodeCount; k++)
        {
            NN->Layers[n].NodeBiases[k] = NN__GetRandomValue();
            NN->Layers[n].NodeValues[k] = NN__GetRandomValue();
            for (int i = 0; i < PrevNodeCount; i++)
            {
                NN->Layers[n].Weights[k*PrevNodeCount + i] = NN__GetRandomValue();
            }
        }

        PrevNodeCount = NN->Layers[n].NodeCount;
    }
}

static float NN__GetRandomValue(void)
{
    return (float)rand() / RAND_MAX;
}

static float NN__StepFn(float Bias, float Value)
{
    if (Value > Bias)
        return 1.0;
    return 0;
}

#endif
