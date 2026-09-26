
#include <stdint.h>

#define NNFXP_IMPLEMENTATION
#include "nnfxp.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>


static float s_LearningRate = 0.5;
static float s_L2Lambda = 0;
static bool s_Correct = false;
static int s_CorrectCount = 0;
static int s_TotalTrails = 0;
static int s_CurrentDataset = 0;
static nnfxp_type s_InputTrainingData[][2] = {
    {0, 0},
    {0, NNFXP_ONE},
    {NNFXP_ONE, 0},
    {NNFXP_ONE, NNFXP_ONE},
};
static nnfxp_type s_ExpectedOutput[] = {
    0, 
    NNFXP_ONE, 
    NNFXP_ONE, 
    0
};



static void DisplayStats(nnfxp *NN, bool IsTraining)
{
    printf("\nIsTraining:      %s\n", IsTraining? "true" : "false");
    printf("Learning rate:     %f\n", s_LearningRate);
    printf("L2 Lambda:         %f\n", s_L2Lambda);
    printf("Loss:              %f\n", NNFXP_FLT(Nnfxp_CalcLoss(NN, &s_ExpectedOutput[s_CurrentDataset], 1, 0)));
    printf("Current dataset:  [%g %g | %g]\n", 
        NNFXP_FLT(s_InputTrainingData[s_CurrentDataset][0]), 
        NNFXP_FLT(s_InputTrainingData[s_CurrentDataset][1]), 
        NNFXP_FLT(s_ExpectedOutput[s_CurrentDataset])
    );
    printf("Neural network states: \n");
    Nnfxp_Print(NN);
    printf("Correct/Attempts: %d/%d (%g%%)\n", s_CorrectCount, s_TotalTrails, (double)s_CorrectCount / s_TotalTrails * 100);
    printf("Was correct:      %s\n", s_Correct? "true" : "false");

    for (int i = 0; i < 4; i++)
    {
        Nnfxp_FeedForward(NN, &(nnfxp_feedforward_config) {
            .InputCount = 2,
            .Inputs = s_InputTrainingData[i], 
        });
        printf("%g ^ %g = %f\n", 
            NNFXP_FLT(s_InputTrainingData[i][0]), 
            NNFXP_FLT(s_InputTrainingData[i][1]), 
            NNFXP_FLT(Nnfxp_GetOutputs(NN)[0])
        );
    }
    printf("\n");
}

static void DoTraining(nnfxp *NN, bool IsTraining)
{
    Nnfxp_FeedForward(NN, &(nnfxp_feedforward_config) {
        .InputCount = 2,
        .Inputs = s_InputTrainingData[s_CurrentDataset],
    });
    if (IsTraining)
    {
        Nnfxp_Backprop(NN, &(nnfxp_backprop_config) {
            .ExpectedOutputCount = 1,
            .ExpectedOutputs = &s_ExpectedOutput[s_CurrentDataset],
            .LearningRate = NNFXP(s_LearningRate),
            .L2Lambda = NNFXP(s_L2Lambda),
        });
    }

    s_TotalTrails++;
    s_Correct = (abs((Nnfxp_GetOutputs(NN)[0] - s_ExpectedOutput[s_CurrentDataset])) < NNFXP(0.1));
    if (s_Correct)
    {
        s_CorrectCount++;
    }
}

int main(void)
{
    srand(time(NULL));
    int NodeCounts[] = { 2, 1 };
    nnfxp NN = { 0 };
    Nnfxp_Create(&NN, &(nnfxp_config) {
        .InputCount = 2,
        .LayerCount = 2,
        .NodeCountPerLayer = NodeCounts,
    });

    bool IsTraining = true;
    while (1)
    {
        printf("\n> ");
        char Input = getc(stdin);

        switch (Input)
        {
        case 'q':
            return 0;

        case '0':
        case '1':
        case '2':
        case '3':
        {
            printf("Using dataset #%c\n", Input);
            s_CurrentDataset = Input - '0';
        } break;
        case 'D':
        {
            DisplayStats(&NN, IsTraining);
        } break;
        case 'T':
        {
            IsTraining = !IsTraining;
        } break;
        case 't':
        {
            DoTraining(&NN, IsTraining);
            DisplayStats(&NN, IsTraining);
        } break;
        case 'r':
        {
            Nnfxp_Randomize(&NN);
        } break;
        case 'R':
        {
            s_TotalTrails = 0;
            s_CorrectCount = 0;
            for (int i = 0; i < 10000; i++)
            {
                s_CurrentDataset = ((float)rand() / RAND_MAX * 4);
                DoTraining(&NN, IsTraining);
                DisplayStats(&NN, IsTraining);
                usleep(100);
            }
        } break;
        case '\n':
            break;
        default:
        {
            s_CurrentDataset = ((float)rand() / RAND_MAX * 4);
            DoTraining(&NN, IsTraining);
            DisplayStats(&NN, IsTraining);
        } break;
        }
    }

    Nnfxp_Destroy(&NN);
    return 0;
}
