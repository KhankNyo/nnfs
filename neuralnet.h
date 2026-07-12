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


static void *NN__DefaultAllocatorCallback(void *Data, neuralnet_allocator_param *Param);


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
        int PrevLayerNodeCount = Config->InputCount;
        for (int i = 0; i < Config->LayerCount; i++)
        {
            int CurrLayerNodeCount = Config->NodeCountPerLayer[i];
            int WeightCount = PrevLayerNodeCount*CurrLayerNodeCount;
            int BiasCount = CurrLayerNodeCount;

            /* TODO: allocate these in order */
            sizeof(NN.Layers[0]);
            WeightCount*sizeof(NN.Layers[0].Weights[0]);
            BiasCount*sizeof(NN.Layers[0].NodeBiases[0]);
            CurrLayerNodeCount*sizeof(NN.Layers[0].NodeValues[0]);

            PrevLayerNodeCount = CurrLayerNodeCount;
        }
    }
    return NN;
}

void NeuralNet_FeedForward(neuralnet *NN, neuralnet_activation_fn Fn)
{
#if 0
    for each level:
        for each node:
            value = accumulate prev node * weights to this node
            node true value = activation_fn(bias, value);
#endif
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

#endif
