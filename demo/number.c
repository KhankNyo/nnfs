
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <time.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "extern/stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "extern/stb_image.h"

#define IMAGE_IMPLEMENTATION
#include "extern/image.h"

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
#define ASSERTF(x, ...) do {\
    if (!(x)) {\
        printf("Assertion failed on line %d in %s:\n", __LINE__, __FILE__);\
        printf("    "#x);\
        printf("\n"__VA_ARGS__);\
        abort();\
    }\
} while (0)


typedef enum 
{
    PREDICT_FLAG_NONE = 0,
    PREDICT_FLAG_ENABLE_BACKPROP = 1 << 0,
    PREDICT_FLAG_RGBA_IMAGE = 1 << 1,
} predict_flags;

typedef struct
{
    int Count;
    int *Labels;
    uint8_t *Samples;
    void *Arena; /* only call free on arena since it owns the memory of Labels and Data */
} data;

typedef struct
{
    predict_flags Flags;

    float LearningRate;
    float L2Lambda;
    const uint8_t *Image;
    int Label;

    float *OutLoss;
} predict_params;

typedef struct
{
    float FalseNegativeThreshold;
    float TruePositiveThreshold;
    int TruePositiveCount;

    int SampleCount;
    int *Labels;
    float *Loss;
} verdict_config;


static float g_FpNNInputs[IMAGE_PIXEL_COUNT];
static float g_FpNNExpectedOutputs[DIGIT_COUNT];


static void *AllocateMemory(size_t ByteCount)
{
    void *Ptr = malloc(ByteCount);
    ASSERTF(Ptr, "Out of memory trying to allocate %fkb.", (float)ByteCount / KB);
    return Ptr;
}


static data LoadTrainingCSV(const char *FileName, int SampleCount, int ImageWidth, int ImageHeight)
{
    data Data = { 0 };
    int _;
    (void)_;

    /* terrible code for loading the training data but idgaf */
    FILE *TrainingFile = fopen(FileName, "rb");
    ASSERTF(TrainingFile, "Unable to open %s\n", FileName);
    {
        int AllocSize = SampleCount
            * (sizeof(Data.Labels[0]) + ImageWidth*ImageHeight);
        Data.Arena = AllocateMemory(AllocSize);
        Data.Labels = Data.Arena;
        Data.Samples = (void *)(Data.Labels + SampleCount);

        uint8_t *Ptr = Data.Samples;
        while (!feof(TrainingFile) && Data.Count < SampleCount)
        {
            int Label = 0;
            _ = fscanf(TrainingFile, "%d,", &Label);
            Data.Labels[Data.Count] = Label;

            for (int y = 0; y < IMAGE_HEIGHT; y++)
            {
                for (int x = 0; x < IMAGE_WIDTH; x++)
                {
                    int Pixel = 0;
                    _ = fscanf(TrainingFile, "%d,", &Pixel);
                    uint8_t PixelByte = Pixel;
                    memcpy(Ptr, &PixelByte, TRAINING_IMAGE_CHANNEL_COUNT);
                    Ptr += TRAINING_IMAGE_CHANNEL_COUNT;
                }
            }
            _ = fscanf(TrainingFile, "\n");

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

static bool Predict(neuralnet *NN, predict_params *Params)
{
    if (Params->Flags & PREDICT_FLAG_RGBA_IMAGE)
    {
        for (int i = 0; i < IMAGE_PIXEL_COUNT; i++)
        {
            g_FpNNInputs[i] = Params->Image[i*4] * (1.0 / 255.0);
        }
    }
    else
    {
        /* straightforward for the compiler to do simd optimization, can't be bothered */
        for (int i = 0; i < IMAGE_PIXEL_COUNT; i++)
        {
            g_FpNNInputs[i] = Params->Image[i] * (1.0 / 255.0); /* normalizing color channel from 0..255 to 0..1 */
        }
    }

    NeuralNet_FeedForward(NN, &(neuralnet_feedforward_config) {
        .InputCount = IMAGE_PIXEL_COUNT,
        .Inputs = g_FpNNInputs,
    });

    memset(g_FpNNExpectedOutputs, 0, sizeof g_FpNNExpectedOutputs);
    g_FpNNExpectedOutputs[Params->Label] = 1.0;
    if (Params->Flags & PREDICT_FLAG_ENABLE_BACKPROP)
    {
        NeuralNet_Backprop(NN, &(neuralnet_backprop_config) {
            .ExpectedOutputCount = DIGIT_COUNT,
            .ExpectedOutputs = g_FpNNExpectedOutputs,
            .LearningRate = Params->LearningRate,
            .L2Lambda = Params->L2Lambda,
        });
    }
    if (Params->OutLoss)
    {
        *Params->OutLoss = NeuralNet_CalcLoss(NN, g_FpNNExpectedOutputs, DIGIT_COUNT, Params->L2Lambda);
    }

    const float *Outputs = NeuralNet_GetOutput(NN);
    bool IsCorrect = FindMaxIndex(Outputs, DIGIT_COUNT) == Params->Label;
    return IsCorrect;
}

static void PrintVerdict(neuralnet *NN, const verdict_config *Config)
{
    const float *Output = NeuralNet_GetOutput(NN);

    if (Config->SampleCount == 1)
    {
        printf("Best guess: %d\n", FindMaxIndex(Output, DIGIT_COUNT));
        printf("Expected:    [");
        for (int i = 0; i < DIGIT_COUNT; i++)
            printf("%4.3f ", g_FpNNExpectedOutputs[i]);
        printf("]\n");

        printf("Digits 0..9: [");
        for (int i = 0; i < DIGIT_COUNT; i++)
            printf("%4.3f ", Output[i]);
        printf("]\n");
        printf("Correctness: [");
        for (int i = 0; i < DIGIT_COUNT; i++)
        {
            if (i == Config->Labels[Config->SampleCount - 1])
                printf("  %c   ", Output[i] > Config->TruePositiveThreshold? 'o' : 'X');
            else
                printf("  %c   ", Output[i] < Config->FalseNegativeThreshold? '_' : 'x');
        }
        printf("]\n");
    }
    float Precision = (float)Config->TruePositiveCount / (Config->SampleCount);
    printf("Precision: %4.2f%% (%d/%d)\n", Precision * 100, Config->TruePositiveCount, Config->SampleCount);

    /* param stats */
    {
        neuralnet_param_stats Stats = NeuralNet_GetParamStats(NN);
        printf("wmin: %f, wmax: %f\n", 
            Stats.WeightMin,
            Stats.WeightMax
        );
    }

    /* calc avg loss */
    {
        int LabelCounts[DIGIT_COUNT] = { 0 };
        float AvgLosses[DIGIT_COUNT] = { 0 };
        for (int i = 0; i < Config->SampleCount; i++)
        {
            int Label = Config->Labels[i];
            float *LabelLoss = &AvgLosses[Label];
            int *LabelCount = &LabelCounts[Label];

            *LabelLoss += Config->Loss[i];
            *LabelCount += 1;
        }
        for (int i = 0; i < DIGIT_COUNT; i++)
        {
            if (LabelCounts[i] > 1)
                AvgLosses[i] /= (float)LabelCounts[i];
        }

        if (Config->SampleCount == 1)
        {
            printf("Loss = %f\n", AvgLosses[Config->Labels[0]]);
        }
        else
        {
            printf("AvgLoss:     [");
            for (int i = 0; i < DIGIT_COUNT; i++)
            {
                printf("%4.3f ", AvgLosses[i]);
            }
            printf("]\n");
        }
    }
    printf("----------------------------\n");
}

static void WriteTestSampleToFile(const char *FileName, const uint8_t *Data)
{
    uint32_t *Image = AllocateMemory(IMAGE_PIXEL_COUNT*sizeof(Image[0]));
    for (int i = 0; i < IMAGE_PIXEL_COUNT; i++)
    {
        Image[i] = 0xFF000000 | Data[i] | (uint32_t)Data[i] << 8 | (uint32_t)Data[i] << 16;
    }

    stbi_write_bmp(FileName, IMAGE_WIDTH, IMAGE_HEIGHT, 4, Image);
    free(Image);
}

static void InputValue(const char *What, float *Value)
{
    float Input = 0;
    int _ = fscanf(stdin, "%f", &Input);
    (void)_;
    printf("%s changed from %f to %f\n", What, *Value, Input);
    *Value = Input;
}


int main(int ArgumentCount, char **Arguments)
{
    (void)ArgumentCount, (void)Arguments;
    srand(time(NULL)); /* uncomment for random weight and bias initialization values */

    /* https://github.com/phoebetronic/mnist/tree/main */
    const char *TrainingFileName = "mnist_train.csv";
    const char *TestingFileName = "mnist_test.csv";
    const char *RandomPredictionFileName = "p.bmp";
    const char *InputFileName = "input";

    /* config */
    float TruePositiveThreshold = 0.7;
    float FalseNegativeThreshold = 0.3;
    float LearningRate = 0.5;
    float L2Lambda = 0.7;
#if 1
#define MODEL_LAYER_COUNT 2
    int ModelArchitectureBuzzword[MODEL_LAYER_COUNT] = {
        [0] = 64,
        [1] = DIGIT_COUNT, /* output layer */
    };
#else
#define MODEL_LAYER_COUNT 3
    int ModelArchitectureBuzzword[MODEL_LAYER_COUNT] = {
        [0] = 128,
        [1] = 64,
        [2] = DIGIT_COUNT, /* output layer */
    };
#endif

    int TrainingSampleCount = 60000; /* mnist_train.csv contains 60k training samples */
    int TestingSampleCount = 10000;
    data TrainingData = { 0 };
    data TestingData = { 0 };
    float *TrainingLoss = NULL;
    float *TestingLoss = NULL;
    {
        printf("Loading training data...\n");
        TrainingData = LoadTrainingCSV(TrainingFileName, TrainingSampleCount, IMAGE_WIDTH, IMAGE_HEIGHT);
        printf("%d training samples loaded\n", TrainingSampleCount);
        printf("Loading testing data...\n");
        TestingData = LoadTrainingCSV(TestingFileName, TestingSampleCount, IMAGE_WIDTH, IMAGE_HEIGHT);
        printf("%d testing samples loaded\n", TestingSampleCount);

        TrainingLoss = AllocateMemory(TrainingSampleCount * sizeof(TrainingLoss[0]));
        TestingLoss = AllocateMemory(TestingSampleCount * sizeof(TestingLoss[0]));

        neuralnet NN = NeuralNet_Create(&(neuralnet_config) {
            .InputCount = IMAGE_PIXEL_COUNT,
            .LayerCount = MODEL_LAYER_COUNT,
            .NodeCountPerLayer = ModelArchitectureBuzzword,
        });
        printf("try 'h' for help\n");
        while (1)
        {
            printf("\n> ");
            char Input = getc(stdin);

            switch (Input)
            {
            case '\n': 
                break;
            default:
            {
                printf("uwotm8, try 'h' for help\n");
            } break;
            case 'h':
            {
                printf(
                    "    q    - quit\n"
                    "    i[n] - predict from given image with the name '%s[n].png',\n"
                    "             ex: 'input0.png' for image with number 0, command: 'i0'.\n"
                    "    T    - train all from training sample '%s'\n"
                    "    p    - predict a random sample from test suite '%s'\n"
                    "    P    - predict all from test suite '%s'\n"
                    "    r    - Reset all weights\n"
                    "    l[f] - Set learning rate\n"
                    "             ex: 'l0.1'\n"
                    "    y[f] - Set L2 regularization lambda\n"
                    "             ex: 'y0.1'\n"
                    "    L    - Display learning rate\n"
                    "    Y    - Display L2 regularization lambda\n",
                    InputFileName,
                    TrainingFileName,
                    TestingFileName,
                    TestingFileName
                );
            } break;

            case 'q':
                goto Out;

            case 'r':
            {
                NeuralNet_Randomize(&NN);
                printf("Neural network randomized.\n");
            } break;

            case 'L':
            {
                printf("Learning rate: %f\n", LearningRate);
            } break;
            case 'Y':
            {
                printf("L2 lambda: %f\n", L2Lambda);
            } break;
            case 'l':
            {
                InputValue("Learning rate", &LearningRate);
            } break;
            case 'y':
            {
                InputValue("L2 lambda", &L2Lambda);
            } break;

            case 'T': /* train all (training dataset) */
            {
                int TruePositiveCount = 0;
                double Start = clock();
                {
                    for (int i = 0; i < TrainingSampleCount; i++)
                    {
                        const uint8_t *Sample = TrainingData.Samples + i*IMAGE_PIXEL_COUNT;
                        int Digit = TrainingData.Labels[i];
                        printf("\rTraining in progress: %d/%d, precision: %4.2f%% (%d/%d)", 
                            i, TrainingSampleCount, (float)TruePositiveCount / (i + 1) * 100, TruePositiveCount, TrainingSampleCount
                        );
                        bool Correct = Predict(&NN, &(predict_params) {
                            .Flags = PREDICT_FLAG_ENABLE_BACKPROP,
                            .LearningRate = LearningRate, 
                            .Image = Sample,
                            .Label = Digit, 
                            .L2Lambda = L2Lambda / TrainingSampleCount,

                            .OutLoss = &TrainingLoss[i],
                        });
                        TruePositiveCount += Correct;
                    }
                }
                double Dt = (clock() - Start) / CLOCKS_PER_SEC;
                printf("\ntime: %fs\n", Dt);

                PrintVerdict(&NN, &(verdict_config) {
                    .TruePositiveCount = TruePositiveCount,
                    .FalseNegativeThreshold = FalseNegativeThreshold,
                    .TruePositiveThreshold = TruePositiveThreshold,

                    .SampleCount = TrainingSampleCount,
                    .Labels = TrainingData.Labels,
                    .Loss = TrainingLoss,
                });
            } break;
            case 'p': /* predict a random sample from test suite */
            {
                int Index = (float)rand() / RAND_MAX * (TestingSampleCount - 1);
                int Digit = TestingData.Labels[Index];
                const uint8_t *Sample = TestingData.Samples + Index*IMAGE_PIXEL_COUNT;

                WriteTestSampleToFile(RandomPredictionFileName, Sample);
                printf("Wrote test sample to '%s'\n", RandomPredictionFileName);

                float Loss = 0;
                bool Correct = Predict(&NN, &(predict_params) {
                    .LearningRate = LearningRate, 
                    .Image = Sample, 
                    .Label = Digit,
                    .L2Lambda = L2Lambda / TestingSampleCount,

                    .OutLoss = &Loss,
                });
                PrintVerdict(&NN, &(verdict_config) {
                    .TruePositiveCount = Correct,
                    .FalseNegativeThreshold = FalseNegativeThreshold,
                    .TruePositiveThreshold = TruePositiveThreshold,

                    .SampleCount = 1,
                    .Labels = &Digit,
                    .Loss = &Loss,
                });
            } break;
            case 'P': /* predict all (testing dataset) */
            {
                int TruePositiveCount = 0;
                for (int i = 0; i < TestingSampleCount; i++)
                {
                    const uint8_t *Sample = TestingData.Samples + i*IMAGE_PIXEL_COUNT;
                    int Digit = TestingData.Labels[i];

                    printf("Testing in progress: %d/%d, precision: %4.2f%%\r", i, TestingSampleCount, (float)TruePositiveCount / (i + 1) * 100);
                    TruePositiveCount += Predict(&NN, &(predict_params) {
                        .LearningRate = LearningRate, 
                        .Image = Sample, 
                        .Label = Digit,
                        .L2Lambda = L2Lambda / TestingSampleCount,

                        .OutLoss = &TestingLoss[i],
                    });
                }
                PrintVerdict(&NN, &(verdict_config) {
                    .TruePositiveCount = TruePositiveCount,
                    .FalseNegativeThreshold = FalseNegativeThreshold,
                    .TruePositiveThreshold = TruePositiveThreshold,

                    .SampleCount = TestingSampleCount,
                    .Labels = TestingData.Labels,
                    .Loss = TestingLoss,
                });
            } break;
            case 'i': /* input image */
            {
                int Digit = getc(stdin) - '0';

                int RequiredChannels = 4;
                int Width = 0, Height = 0, Channels = 0;
                static char TmpFileName[128];
                snprintf(TmpFileName, sizeof TmpFileName, "%s%d.png", InputFileName, Digit);
                uint8_t *Data = stbi_load(TmpFileName, &Width, &Height, &Channels, RequiredChannels);
                if (Data)
                {
                    if (Width != IMAGE_WIDTH && Height != IMAGE_HEIGHT)
                    {
                        /* resize image */
                        uint8_t *NewData = AllocateMemory(IMAGE_PIXEL_COUNT*RequiredChannels);

                        Image_Resize(&(image_resize_config) {
                            .Dst = NewData,
                            .DstWidth = IMAGE_WIDTH,
                            .DstHeight = IMAGE_HEIGHT,
                            .Src = Data,
                            .SrcWidth = Width, 
                            .SrcHeight = Height,
                            .Format = IMAGE_PIXEL_FORMAT_RGBA32,
                            .Method = IMAGE_RESIZE_METHOD_NEAREST_NEIGHBOR,
                        });

                        free(Data);
                        Width = IMAGE_WIDTH;
                        Height = IMAGE_HEIGHT;
                        Data = NewData;
                    }

                    float Loss = 0;
                    bool IsCorrect = Predict(&NN, &(predict_params) {
                        .Flags = PREDICT_FLAG_RGBA_IMAGE | PREDICT_FLAG_ENABLE_BACKPROP,

                        .LearningRate = LearningRate, 
                        .Image = Data, 
                        .Label = Digit,
                        .L2Lambda = L2Lambda / TrainingSampleCount,

                        .OutLoss = &Loss,
                    });
                    PrintVerdict(&NN, &(verdict_config) {
                        .TruePositiveCount = IsCorrect,
                        .FalseNegativeThreshold = FalseNegativeThreshold,
                        .TruePositiveThreshold = TruePositiveThreshold,

                        .SampleCount = 1,
                        .Labels = &Digit,
                        .Loss = &Loss,
                    });

                    /* write out training image (diff name) since it could've been resized */
                    stbi_write_bmp(RandomPredictionFileName, Width, Height, Channels, Data);
                    free(Data);
                }
                else
                {
                    printf("Unable to load '%s'\n", TmpFileName);
                }
            } break;
            }
        }
Out:
        NeuralNet_Destroy(&NN);
    }
    free(TrainingData.Arena);
    free(TestingData.Arena);
    free(TrainingLoss);
    free(TestingLoss);
    return 0;
}

