#ifndef NEURALNET_H
#define NEURALNET_H

#include <stdbool.h>

typedef struct neuralnet neuralnet;
typedef struct neuralnet_layer neuralnet_layer;
typedef struct neuralnet_config neuralnet_config;
typedef struct neuralnet_backprop_config neuralnet_backprop_config;
typedef struct neuralnet_delta_i_param neuralnet_delta_i_param;
typedef float (*neuralnet_activation_fn)(float Value);
typedef float (*neuralnet_delta_i_fn)(neuralnet *NN, neuralnet_delta_i_param *Param);

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
    neuralnet_delta_i_fn DeltaIFn;
    float LearningRate;
    int ExpectedOutputCount;
    const float *ExpectedOutputs;
};

struct neuralnet_delta_i_param
{
    bool IsOutputLayer;
    float NodeValue;
    union {
        struct {
            float ExpectedValue;
        } OutputLayer;
        struct {
            float NextLayerDeltaIValue;
            float WeightOrBias;
        } InnerLayer;
    };
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
    /* [array of OutputCount biases and values] */
    float *NodeBiases;
    float *NodeValues;
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

#define NN__ALLOCATION_SCOPE_BEGIN(p_nn)
#define NN__ALLOCATION_SCOPE_END(p_nn)

static void *NN__DefaultAllocatorCallback(void *Data, neuralnet_allocator_param *Param);
static void NN__LinearCombination(float *Y, const float *Mat, const float *X, const float *B, int NumRow, int NumCol);
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
    /* NOTE: NeuralNet_Init() guarantees that 
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
#if 0
    for each layer: 
        for all weights of each layer: 
            calc new values:
                calc delta_i:
                    EXAMPLE pseudocode for delta_i using cost_fn(output) = 0.5 * sum((expected_i - output_i)^2), 
                                                         activation_fn(x) = 1 / (1 + e^(-x)):
                    d_sigmoid = node_value*(1 - node_value);
                    if (is_output_layer)
                        delta_i = d_sigmoid * (expected - node_value);      // NOTE: node_value == output, also note that cost_fn() turned into only 'expected - node_value' because of partial derivative with respect to 'node_value'
                    else
                        delta_i = d_sigmoid * (weight_or_bias * next_layer_delta_i);
                calc change in value: 
                    d_value = learning_rate * delta_i * prev_node_value     // NOTE: this is NODE VALUE, 
                                                                            //  not the value of the weight or biases, but the output value of 
                                                                            //  the node connecting to the node we came from
                calc value (weight/biases):
                    value += d_value                                        // NOTE: sum the values so the changes can accumulate

        same sequence for biases
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

/* NOTE: Y = (M^T) . X + B
 * [Y_0]   [TM_00 TM_0r]   [X_0]   [B_0]
 * [Y_r] = [TM_c0 TM_cr] . [X_c] + [B_c]
 */
static void NN__LinearCombination(float *Y, const float *TranposedMat, const float *X, const float *B, int NumRow, int NumCol)
{
    for (int r = 0; r < NumRow; r++)
    {
        float Tmp = 0;
        for (int c = 0; c < NumCol; c++)
        {
            /* NOTE: linear index for cache locality */
            Tmp += TranposedMat[c + r*NumRow] * X[c] + B[c];
        }
        Y[r] = Tmp;
    }
}

/* NOTE: Y = A . B^T, 
 * NumColB is the number of columns in B (aka (B^T)^T) */
static void NN__MatDot(float *Y, const float *A, const float *TranposedB, int NumRowA, int NumColA, int NumRowTB)
{
    int NumColB = NumRowTB;
    for (int Cb = 0; Cb < NumColB; Cb++)
    {
        for (int Ra = 0; Ra < NumRowA; Ra++)
        {
            float Tmp = 0;
            for (int Ca = 0; Ca < NumColA; Ca++)
            {
                Tmp += A[Ca + Ra*NumRowA] * TranposedB[Ca + Cb*NumColB];
            }
            Y[Ra*NumRowA + Cb] = Tmp;
        }
    }
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
