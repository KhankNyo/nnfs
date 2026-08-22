#ifndef NNFXP_H
#define NNFXP_H


#include "neuralnet.h"

#ifndef nnfxp_type
#  include <stdint.h>
#  define NNFXP_TYPE_MAX INT8_MAX
#  define NNFXP_TYPE_MIN INT8_MIN
#  define nnfxp_type int8_t
#endif /* nnfxp_type */


typedef struct nnfxp nnfxp;
typedef struct nnfxp_layer nnfxp_layer;
typedef struct nnfxp_feedforward_config nnfxp_feedforward_config;

struct nnfxp_feedforward_config
{
    int InputCount;
    const nnfxp_type *Inputs;
};


nnfxp Nnfxp_CreateFromNeuralNet(const neuralnet *NN);
void Nnfxp_Destroy(nnfxp *NN);
void Nnfxp_FeedForward(nnfxp *NN, const nnfxp_feedforward_config *Config);


struct nnfxp_layer
{
    int InputCount;
    int OutputCount;
    nnfxp_type *Weights;
    nnfxp_type *Biases;
    nnfxp_type *Outputs;
};

struct nnfxp
{
    int DecimalPlaceCount;
    int InputCount;
    int LayerCount;
    nnfxp_type *Inputs;
    nnfxp_layer *Layers;

    void *AllocatorData;
    neuralnet_allocator_callback AllocatorCallback;
};


#endif /* NNFXP */


#if defined(NNFXP_IMPLEMENTATION) && !defined(NNFXP_ALREADY_IMPLEMENTED)
#define NNFXP_ALREADY_IMPLEMENTED

#include <float.h>
#include <stdio.h>


nnfxp Nnfxp_CreateFromNeuralNet(const neuralnet *NN)
{
    nnfxp Nnfxp = { 0 };
    return Nnfxp;
}

void Nnfxp_Destroy(nnfxp *NN)
{
}

void Nnfxp_FeedForward(nnfxp *NN, const nnfxp_feedforward_config *Config)
{
}


#endif
