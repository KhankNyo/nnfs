
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <time.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "extern/stb_image_write.h"

#define NEURALNET_IMPLEMENTATION
#include "neuralnet.h"

#define IMAGE_WIDTH 28
#define IMAGE_HEIGHT 28
#define IMAGE_CHANNEL_COUNT 4
#define TRAINING_IMAGE_CHANNEL_COUNT 1
#define IMAGE_PIXEL_COUNT (IMAGE_WIDTH*IMAGE_HEIGHT)
#define DIGIT_COUNT 10
#define KB 1024
#define MODEL_LAYER_COUNT 3
#define ASSERTF(x, ...) do {\
    if (!(x)) {\
        printf("Assertion failed on line %d in %s:\n", __LINE__, __FILE__);\
        printf("    "#x);\
        printf("\n"__VA_ARGS__);\
        abort();\
    }\
} while (0)

typedef struct
{
    int Count;
    int *Labels;
    uint8_t *Samples;
    void *Arena; /* only call free on arena since it owns the memory of Labels and Data */
} data;

typedef enum 
{
    PREDICT_FLAG_NONE = 0,
    PREDICT_FLAG_ENABLE_BACKPROP = 1 << 0,
} predict_flags;


static int s_TruePositiveCount = 0;


static data LoadTrainingCSV(const char *FileName, int SampleCount, int ImageWidth, int ImageHeight)
{
    data Data = { 0 };

    /* terrible code for loading the training data but idgaf */
    FILE *TrainingFile = fopen(FileName, "rb");
    ASSERTF(TrainingFile, "Unable to open %s\n", FileName);
    {
        int AllocSize = SampleCount
            * (sizeof(Data.Labels[0]) + ImageWidth*ImageHeight);
        Data.Arena = malloc(AllocSize);
        ASSERTF(Data.Arena, "Out of memory trying to allocate %dkb\n", AllocSize / KB);

        Data.Labels = Data.Arena;
        Data.Samples = (void *)(Data.Labels + SampleCount);

        uint8_t *Ptr = Data.Samples;
        while (!feof(TrainingFile) && Data.Count < SampleCount)
        {
            int Label = 0;
            fscanf(TrainingFile, "%d,", &Label);
            Data.Labels[Data.Count] = Label;

            for (int y = 0; y < IMAGE_HEIGHT; y++)
            {
                for (int x = 0; x < IMAGE_WIDTH; x++)
                {
                    int Pixel = 0;
                    fscanf(TrainingFile, "%d,", &Pixel);
                    uint8_t PixelByte = Pixel;
                    memcpy(Ptr, &PixelByte, TRAINING_IMAGE_CHANNEL_COUNT);
                    Ptr += TRAINING_IMAGE_CHANNEL_COUNT;
                }
            }
            fscanf(TrainingFile, "\n");

            Data.Count++;
        }
    }
    fclose(TrainingFile);
    return Data;
}

static void Predict(neuralnet *NN, float LearningRate, const uint8_t *Image, int ExpectedDigit, predict_flags Flags)
{
    static float FpNNInputs[IMAGE_PIXEL_COUNT];
    for (int i = 0; i < IMAGE_PIXEL_COUNT; i++)
    {
        FpNNInputs[i] = Image[i] * (1.0 / 255.0); /* normalizing color channel from 0..255 to 0..1 */
    }

    NeuralNet_FeedForward(NN, &(neuralnet_feedforward_config) {
        .InputCount = IMAGE_PIXEL_COUNT,
        .Inputs = FpNNInputs,
    });


    if (Flags & PREDICT_FLAG_ENABLE_BACKPROP)
    {
        static float FpNNExpectedOutputs[DIGIT_COUNT];
        memset(FpNNExpectedOutputs, 0, sizeof FpNNExpectedOutputs);
        FpNNExpectedOutputs[ExpectedDigit] = 1.0;
        NeuralNet_Backprop(NN, &(neuralnet_backprop_config) {
            .ExpectedOutputCount = DIGIT_COUNT,
            .ExpectedOutputs = FpNNExpectedOutputs,
            .LearningRate = LearningRate,
        });

        printf("Expected:    [");
        for (int i = 0; i < DIGIT_COUNT; i++)
            printf("%4.3f ", FpNNExpectedOutputs[i]);
        printf("\n");
    }

}

static void PrintVerdict(neuralnet *NN, int TotalSample, int Expected, float FalseNegativeThreshold, float TruePositiveThreshold)
{
    const float *Output = NeuralNet_GetOutput(NN);
    s_TruePositiveCount += Output[Expected] > TruePositiveThreshold;

    printf("Digits 0..9: [");
    for (int i = 0; i < DIGIT_COUNT; i++)
        printf("%4.3f ", Output[i]);
    printf("]\n");
    printf("Correctness: [");
    for (int i = 0; i < DIGIT_COUNT; i++)
    {
        if (i == Expected)
            printf("  %c   ", Output[i] > TruePositiveThreshold? 'o' : 'X');
        else
            printf("  %c   ", Output[i] < FalseNegativeThreshold? '_' : 'x');
    }
    printf("]\n");
    float Precision = (float)s_TruePositiveCount / (TotalSample);
    printf("Precision: %4.2f%% (%d/%d)\n", Precision * 100, s_TruePositiveCount, TotalSample);
    printf("----------------------------\n");
}


int main(int ArgumentCount, char **Arguments)
{
    srand(time(NULL));
    /* https://github.com/phoebetronic/mnist/tree/main */
    const char *TrainingFileName = "mnist_train.csv";
    const char *TestFileName = "mnist_test.csv";

    /* config */
    float TruePositiveThreshold = 0.8;
    float FalseNegativeThreshold = 0.2;
    float LearningRate = 1.0;
    // https://www.geeksforgeeks.org/machine-learning/handwritten-digit-recognition-using-neural-network/
    int ModelArchitectureBuzzword[MODEL_LAYER_COUNT] = {
        [0] = 128,
        [1] = 64,
        [2] = DIGIT_COUNT, /* output layer */
    };

    int TrainingSampleCount = 60000; /* mnist_train.csv contains 60k training samples */
    data TrainingData = { 0 };
    {
        printf("Loading training data...\n");
        TrainingData = LoadTrainingCSV(TrainingFileName, TrainingSampleCount, IMAGE_WIDTH, IMAGE_HEIGHT);
        printf("%d training samples loaded\n", TrainingSampleCount);

        neuralnet NN = NeuralNet_Create(&(neuralnet_config) {
            .InputCount = IMAGE_PIXEL_COUNT,
            .LayerCount = MODEL_LAYER_COUNT,
            .NodeCountPerLayer = ModelArchitectureBuzzword,
        });
        while (1)
        {
            printf("\n> ");
            char Input = getc(stdin);

            switch (Input)
            {
            default:
            {
                printf("uwotm8, try 'h' for help\n");
            } break;
            case 'h':
            {
                printf(
                    "q - quit\n"
                    "T - train all from training sample '%s'\n"
                    "p - predict a random sample from test suite '%s'\n"
                    "P - predict all from test suite '%s'\n",
                    TrainingFileName,
                    TestFileName, TestFileName
                );
            } break;
            case 'q':
                return 0;
            case 'T': /* train all */
            {
                s_TruePositiveCount = 0;
                for (int i = 0; i < TrainingSampleCount; i++)
                {
                    uint8_t *Sample = TrainingData.Samples + i*IMAGE_PIXEL_COUNT;
                    int Digit = TrainingData.Labels[i];
                    printf("Sample %d: label: %d\n", i, Digit);
                    Predict(&NN, LearningRate, Sample, Digit, PREDICT_FLAG_ENABLE_BACKPROP);
                    PrintVerdict(&NN, i + 1, Digit, FalseNegativeThreshold, TruePositiveThreshold);

                }
            } break;
            case 'p': /* predict a random sample from test suite */
            {
            } break;
            case 'P': /* predict all */
            {
            } break;
            }
        }
        NeuralNet_Destroy(&NN);
    }
    free(TrainingData.Arena);
    return 0;
}

