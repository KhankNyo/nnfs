
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <time.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "extern/stb_image_write.h"

#define NEURALNET_USE_SIMD
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


static float g_FpNNInputs[IMAGE_PIXEL_COUNT];
static float g_FpNNExpectedOutputs[DIGIT_COUNT];


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

static int FindMaxIndex(const float *Data, int Count)
{
    ASSERTF(Count, "Invalid count: %d\n", Count);
    int MaxIndex = 0;
    for (int i = 1; i < Count; i++)
    {
        if (Data[i] > Data[MaxIndex])
            MaxIndex = i;
    }
    return MaxIndex;
}

static bool Predict(neuralnet *NN, float LearningRate, const uint8_t *Image, int ExpectedDigit, predict_flags Flags, float TruePositiveThreshold)
{
    for (int i = 0; i < IMAGE_PIXEL_COUNT; i++)
    {
        g_FpNNInputs[i] = Image[i] * (1.0 / 255.0); /* normalizing color channel from 0..255 to 0..1 */
    }

    NeuralNet_FeedForward(NN, &(neuralnet_feedforward_config) {
        .InputCount = IMAGE_PIXEL_COUNT,
        .Inputs = g_FpNNInputs,
    });

    memset(g_FpNNExpectedOutputs, 0, sizeof g_FpNNExpectedOutputs);
    g_FpNNExpectedOutputs[ExpectedDigit] = 1.0;
    if (Flags & PREDICT_FLAG_ENABLE_BACKPROP)
    {
        NeuralNet_Backprop(NN, &(neuralnet_backprop_config) {
            .ExpectedOutputCount = DIGIT_COUNT,
            .ExpectedOutputs = g_FpNNExpectedOutputs,
            .LearningRate = LearningRate,
        });
    }
    bool IsCorrect = FindMaxIndex(NeuralNet_GetOutput(NN), DIGIT_COUNT) == ExpectedDigit;
    return IsCorrect;
}

static void PrintVerdict(neuralnet *NN, int TotalSample, int TruePositiveCount, int Expected, float FalseNegativeThreshold, float TruePositiveThreshold)
{
    const float *Output = NeuralNet_GetOutput(NN);
    printf("Expected:    [");
    for (int i = 0; i < DIGIT_COUNT; i++)
        printf("%4.3f ", g_FpNNExpectedOutputs[i]);
    printf("]\n");

    printf("Digits 0..9: [");
    for (int i = 0; i < DIGIT_COUNT; i++)
        printf("%4.3f ", Output[i]);
    printf("], best guess: %d\n", FindMaxIndex(Output, DIGIT_COUNT));
    printf("Correctness: [");
    for (int i = 0; i < DIGIT_COUNT; i++)
    {
        if (i == Expected)
            printf("  %c   ", Output[i] > TruePositiveThreshold? 'o' : 'X');
        else
            printf("  %c   ", Output[i] < FalseNegativeThreshold? '_' : 'x');
    }
    printf("]\n");
    float Precision = (float)TruePositiveCount / (TotalSample);
    printf("Precision: %4.2f%% (%d/%d)\n", Precision * 100, TruePositiveCount, TotalSample);
    printf("----------------------------\n");
}

static void WriteTestSampleToFile(const char *FileName, const uint8_t *Data)
{
    uint32_t *Image = malloc(IMAGE_PIXEL_COUNT*sizeof(Image[0]));
    ASSERTF(Image, "Out of memory trying to allocate %dkb\n", (int)(IMAGE_WIDTH*IMAGE_HEIGHT*sizeof(Image[0]) / KB));
    for (int i = 0; i < IMAGE_PIXEL_COUNT; i++)
    {
        Image[i] = 0xFF000000 | Data[i] | (uint32_t)Data[i] << 8 | (uint32_t)Data[i] << 16;
    }

    stbi_write_bmp(FileName, IMAGE_WIDTH, IMAGE_HEIGHT, 4, Image);
    free(Image);
}


int main(int ArgumentCount, char **Arguments)
{
    srand(time(NULL));
    /* https://github.com/phoebetronic/mnist/tree/main */
    const char *TrainingFileName = "mnist_train.csv";
    const char *TestingFileName = "mnist_test.csv";
    const char *RandomPredictionFileName = "p.bmp";

    /* config */
    float TruePositiveThreshold = 0.7;
    float FalseNegativeThreshold = 0.3;
    float LearningRate = 1.0;
    // https://www.geeksforgeeks.org/machine-learning/handwritten-digit-recognition-using-neural-network/
    int ModelArchitectureBuzzword[MODEL_LAYER_COUNT] = {
        [0] = 128,
        [1] = 64,
        [2] = DIGIT_COUNT, /* output layer */
    };

    int TrainingSampleCount = 60000; /* mnist_train.csv contains 60k training samples */
    int TestingSampleCount = 10000;
    data TrainingData = { 0 };
    data TestingData = { 0 };
    {
        printf("Loading training data...\n");
        TrainingData = LoadTrainingCSV(TrainingFileName, TrainingSampleCount, IMAGE_WIDTH, IMAGE_HEIGHT);
        printf("%d training samples loaded\n", TrainingSampleCount);
        printf("Loading testing data...\n");
        TestingData = LoadTrainingCSV(TestingFileName, TestingSampleCount, IMAGE_WIDTH, IMAGE_HEIGHT);
        printf("%d testing samples loaded\n", TestingSampleCount);

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
                    TestingFileName, 
                    TestingFileName
                );
            } break;

            case 'q':
                goto Out;

            case 'T': /* train all */
            {
                int TruePositiveCount = 0;
                int Digit = 0;
                double Start = clock();
                for (int i = 0; i < TrainingSampleCount; i++)
                {
                    const uint8_t *Sample = TrainingData.Samples + i*IMAGE_PIXEL_COUNT;
                    Digit = TrainingData.Labels[i];
                    printf("\rTraining in progress: %d/%d, precision: %4.2f%%", i, TrainingSampleCount, (float)TruePositiveCount / (i + 1) * 100);
                    TruePositiveCount += Predict(&NN, LearningRate, Sample, Digit, PREDICT_FLAG_ENABLE_BACKPROP, TruePositiveThreshold);
                }
                double Dt = (clock() - Start) / CLOCKS_PER_SEC;
                printf("\ntime: %fs\n", Dt);

                PrintVerdict(&NN, TrainingSampleCount, TruePositiveCount, Digit, FalseNegativeThreshold, TruePositiveThreshold);
            } break;
            case 'p': /* predict a random sample from test suite */
            {
                int Index = (float)rand() / RAND_MAX * (TestingSampleCount - 1);
                int Digit = TestingData.Labels[Index];
                const uint8_t *Sample = TestingData.Samples + Index*IMAGE_PIXEL_COUNT;

                WriteTestSampleToFile(RandomPredictionFileName, Sample);
                printf("Wrote test sample to '%s'\n", RandomPredictionFileName);

                bool Correct = Predict(&NN, LearningRate, Sample, Digit, PREDICT_FLAG_NONE, TruePositiveThreshold);
                PrintVerdict(&NN, 1, Correct, Digit, FalseNegativeThreshold, TruePositiveThreshold);
            } break;
            case 'P': /* predict all */
            {
                int TruePositiveCount = 0;
                int Digit = 0;
                for (int i = 0; i < TestingSampleCount; i++)
                {
                    const uint8_t *Sample = TestingData.Samples + i*IMAGE_PIXEL_COUNT;
                    Digit = TestingData.Labels[i];
                    printf("Testing in progress: %d/%d, precision: %4.2f%%\r", i, TestingSampleCount, (float)TruePositiveCount / (i + 1) * 100);
                    TruePositiveCount += Predict(&NN, LearningRate, Sample, Digit, PREDICT_FLAG_NONE, TruePositiveThreshold);
                }
                PrintVerdict(&NN, TestingSampleCount, TruePositiveCount, Digit, FalseNegativeThreshold, TruePositiveThreshold);
            } break;
            }
        }
Out:
        NeuralNet_Destroy(&NN);
    }
    free(TrainingData.Arena);
    free(TestingData.Arena);
    return 0;
}

