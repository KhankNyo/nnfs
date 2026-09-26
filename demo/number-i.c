
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

#define NNFXP_IMPLEMENTATION
#include "nnfxp.h"

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


static nnfxp_type g_FpNNInputs[IMAGE_PIXEL_COUNT];
static nnfxp_type g_FpNNExpectedOutputs[DIGIT_COUNT];


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

/* NOTE: rgba image */
static void CenterImage(uint8_t *Dst, const uint8_t *Src, int Width, int Height)
{
    int ChannelCount = IMAGE_CHANNEL_COUNT;
    int OffsetX = 0;
    int OffsetY = 0;
    {
        int CenterX = 0;
        int CenterY = 0;
        int Intensity = 0;
        for (int y = 0; y < Height; y++)
        {
            int Col = y*Width;
            for (int x = 0; x < Width; x++)
            {
                int Index = (Col + x)*ChannelCount;
                CenterX += x * Src[Index];
                CenterY += y * Src[Index];
                Intensity += Src[Index];
            }
        }

        CenterX = (float)CenterX / Intensity + 0.5;
        CenterY = (float)CenterY / Intensity + 0.5;
        OffsetX = CenterX - Width/2;
        OffsetY = CenterY - Height/2;
    }

    for (int y = 0; y < Height; y++)
    {
        int SrcY = y + OffsetY;
        if (SrcY >= Height)
            SrcY -= Height;
        if (SrcY < 0)
            SrcY += Height;
        for (int x = 0; x < Width; x++)
        {
            int SrcX = x + OffsetX;
            if (SrcX >= Width)
                SrcX -= Width;
            if (SrcX < 0)
                SrcX += Width;
            int SrcIndex = (SrcY*Width + SrcX)*IMAGE_CHANNEL_COUNT;
            int DstIndex = (y*Width + x)*IMAGE_CHANNEL_COUNT;
            memcpy(Dst + DstIndex, Src + SrcIndex, ChannelCount);
        }
    }
}

static int FindMaxIndex(const nnfxp_type *Data, int Count)
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

static bool Predict(nnfxp *NN, predict_params *Params)
{
    if (Params->Flags & PREDICT_FLAG_RGBA_IMAGE)
    {
        for (int i = 0; i < IMAGE_PIXEL_COUNT; i++)
        {
            g_FpNNInputs[i] = NNFXP(Params->Image[i*4] * (1.0 / 255.0));
        }
    }
    else
    {
        /* straightforward for the compiler to do simd optimization, can't be bothered */
        for (int i = 0; i < IMAGE_PIXEL_COUNT; i++)
        {
            g_FpNNInputs[i] = NNFXP(Params->Image[i] * (1.0 / 255.0)); /* normalizing color channel from 0..255 to 0..1 */
        }
    }

    Nnfxp_FeedForward(NN, &(nnfxp_feedforward_config) {
        .InputCount = IMAGE_PIXEL_COUNT,
        .Inputs = g_FpNNInputs,
    });

    memset(g_FpNNExpectedOutputs, 0, sizeof g_FpNNExpectedOutputs);
    g_FpNNExpectedOutputs[Params->Label] = NNFXP(1.0);
    if (Params->Flags & PREDICT_FLAG_ENABLE_BACKPROP)
    {
        Nnfxp_Backprop(NN, &(nnfxp_backprop_config) {
            .ExpectedOutputCount = DIGIT_COUNT,
            .ExpectedOutputs = g_FpNNExpectedOutputs,
            .LearningRate = NNFXP(Params->LearningRate),
            .L2Lambda = NNFXP(Params->L2Lambda),
        });
    }
    if (Params->OutLoss)
    {
        *Params->OutLoss = NNFXP_FLT(Nnfxp_CalcLoss(NN, 
            g_FpNNExpectedOutputs, 
            DIGIT_COUNT, 
            NNFXP(Params->L2Lambda)
        ));
    }

    const nnfxp_type *Outputs = Nnfxp_GetOutputs(NN);
    bool IsCorrect = FindMaxIndex(Outputs, DIGIT_COUNT) == Params->Label;
    return IsCorrect;
}

static void PrintVerdict(nnfxp *NN, const verdict_config *Config)
{
    const nnfxp_type *Output = Nnfxp_GetOutputs(NN);

    if (Config->SampleCount == 1)
    {
        printf("Best guess: %d\n", FindMaxIndex(Output, DIGIT_COUNT));
        printf("Expected:    [");
        for (int i = 0; i < DIGIT_COUNT; i++)
            printf("%4.3f ", NNFXP_FLT(g_FpNNExpectedOutputs[i]));
        printf("]\n");

        printf("Digits 0..9: [");
        for (int i = 0; i < DIGIT_COUNT; i++)
            printf("%4.3f ", NNFXP_FLT(Output[i]));
        printf("]\n");
        printf("Correctness: [");
        for (int i = 0; i < DIGIT_COUNT; i++)
        {
            if (i == Config->Labels[Config->SampleCount - 1])
                printf("  %c   ", NNFXP_FLT(Output[i]) > Config->TruePositiveThreshold? 'o' : 'X');
            else
                printf("  %c   ", NNFXP_FLT(Output[i]) < Config->FalseNegativeThreshold? '_' : 'x');
        }
        printf("]\n");
    }
    float Precision = (float)Config->TruePositiveCount / (Config->SampleCount);
    printf("Precision: %4.2f%% (%d/%d)\n", Precision * 100, Config->TruePositiveCount, Config->SampleCount);

    /* param stats */
    {
        nnfxp_param_stats Stats = Nnfxp_GetParamStats(NN);
        printf("wmin: %f, wmax: %f, bmin: %f, bmax: %f\n", 
            NNFXP_FLT(Stats.WeightMin),
            NNFXP_FLT(Stats.WeightMax),
            NNFXP_FLT(Stats.BiasMin),
            NNFXP_FLT(Stats.BiasMax)
        );
    }

    /* calc avg loss */
    {
        int LabelCounts[DIGIT_COUNT] = { 0 };
        float AvgLosses[DIGIT_COUNT] = { 0 };
        for (int i = 0; i < Config->SampleCount; i++)
        {
            int Label = Config->Labels[i];
            AvgLosses[Label] += (Config->Loss[i]);
            LabelCounts[Label] += 1;
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
#if 0
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
    uint8_t *CenteredImage = NULL;
    predict_flags InputFlags = PREDICT_FLAG_RGBA_IMAGE;
    {
        printf("Loading training data...\n");
        TrainingData = LoadTrainingCSV(TrainingFileName, TrainingSampleCount, IMAGE_WIDTH, IMAGE_HEIGHT);
        printf("%d training samples loaded\n", TrainingSampleCount);
        printf("Loading testing data...\n");
        TestingData = LoadTrainingCSV(TestingFileName, TestingSampleCount, IMAGE_WIDTH, IMAGE_HEIGHT);
        printf("%d testing samples loaded\n", TestingSampleCount);

        TrainingLoss = AllocateMemory(TrainingSampleCount * sizeof(TrainingLoss[0]));
        TestingLoss = AllocateMemory(TestingSampleCount * sizeof(TestingLoss[0]));
        CenteredImage = AllocateMemory(IMAGE_PIXEL_COUNT*IMAGE_CHANNEL_COUNT);

        nnfxp NN = { 0 };
        Nnfxp_Create(&NN, &(nnfxp_config) {
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
                    "    Y    - Display L2 regularization lambda\n"
                    "    I    - Toggle backpropagation for input prediction\n",
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
                Nnfxp_Randomize(&NN);
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
            case 'I':
            {
                if (InputFlags & PREDICT_FLAG_ENABLE_BACKPROP)
                {
                    InputFlags &= ~PREDICT_FLAG_ENABLE_BACKPROP;
                    printf("Disabled backprop for input images.\n");
                }
                else
                {
                    InputFlags |= PREDICT_FLAG_ENABLE_BACKPROP;
                    printf("Enabled backprop for input images.\n");
                }
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
                            .LearningRate = (LearningRate), 
                            .Image = Sample,
                            .Label = Digit, 
                            .L2Lambda = (L2Lambda / TrainingSampleCount),

                            .OutLoss = &TrainingLoss[i],
                        });
                        TruePositiveCount += Correct;
                    }
                }
                double Dt = (clock() - Start) / CLOCKS_PER_SEC;
                printf("\ntime: %fs\n", Dt);

                PrintVerdict(&NN, &(verdict_config) {
                    .TruePositiveCount = TruePositiveCount,
                    .FalseNegativeThreshold = (FalseNegativeThreshold),
                    .TruePositiveThreshold = (TruePositiveThreshold),

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
                    .LearningRate = (LearningRate), 
                    .Image = Sample, 
                    .Label = Digit,
                    .L2Lambda = (L2Lambda / TestingSampleCount),

                    .OutLoss = &Loss,
                });
                PrintVerdict(&NN, &(verdict_config) {
                    .TruePositiveCount = Correct,
                    .FalseNegativeThreshold = (FalseNegativeThreshold),
                    .TruePositiveThreshold = (TruePositiveThreshold),

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
                        .LearningRate = (LearningRate), 
                        .Image = Sample, 
                        .Label = Digit,
                        .L2Lambda = (L2Lambda / TestingSampleCount),

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

                    CenterImage(CenteredImage, Data, Width, Height);

                    float Loss = 0;
                    bool IsCorrect = Predict(&NN, &(predict_params) {
                        .Flags = InputFlags,

                        .LearningRate = LearningRate, 
                        .Image = CenteredImage, 
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
                    stbi_write_bmp(RandomPredictionFileName, Width, Height, Channels, CenteredImage);
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
        Nnfxp_Destroy(&NN);
    }
    free(CenteredImage);
    free(TrainingData.Arena);
    free(TestingData.Arena);
    free(TrainingLoss);
    free(TestingLoss);
    return 0;
}

