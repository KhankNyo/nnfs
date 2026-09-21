
#include <stdio.h>
#include <assert.h>
#include <stdint.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "extern/stb_image_write.h"

#define NEURALNET_IMPLEMENTATION
#include "neuralnet.h"

#define IMAGE_WIDTH 28
#define IMAGE_HEIGHT 28
#define IMAGE_CHANNEL_COUNT 4
#define TRAINING_IMAGE_CHANNEL_COUNT 1
#define IMAGE_PIXEL_COUNT (IMAGE_WIDTH*IMAGE_HEIGHT)
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
            fscanf(TrainingFile, "%d,", &Data.Labels[Data.Count]);
            printf("Digit: %d\n", Data.Labels[Data.Count]);
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


int main(int ArgumentCount, char **Arguments)
{
    if (ArgumentCount != 2)
    {
        printf("Usage: %s [test image index]\n", Arguments[0]);
        return 1;
    }

    /* https://github.com/phoebetronic/mnist/tree/main */
    const char *TrainingFileName = "mnist_train.csv";
    const char *TestFileName = "mnist_test.csv";
    int TestImageIndex = atoi(Arguments[1]);
    const char *OutputFileName = "test.bmp";

    int TrainingSampleCount = 60000; /* mnist_train.csv contains 60k training samples */
    data TrainingData = { 0 };
    {
        TrainingData = LoadTrainingCSV(TrainingFileName, TrainingSampleCount, IMAGE_WIDTH, IMAGE_HEIGHT);

        // https://www.geeksforgeeks.org/machine-learning/handwritten-digit-recognition-using-neural-network/
        int NodeCountPerLayer[MODEL_LAYER_COUNT] = {
            [0] = 128,
            [1] = 64,
            [2] = 10, /* output layer, number of digits */
        };
        neuralnet NN = NeuralNet_Create(&(neuralnet_config) {
            .InputCount = IMAGE_PIXEL_COUNT,
            .LayerCount = MODEL_LAYER_COUNT,
            .NodeCountPerLayer = NodeCountPerLayer,
        });
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
                for (int i = 0; i < TrainingSampleCount; i++)
                {

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

