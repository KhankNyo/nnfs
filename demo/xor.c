

#define NEURALNET_IMPLEMENTATION
#include "neuralnet.h"

#include <stdio.h>
#include <string.h>


static float s_LearningRate = 1;
static bool s_Correct = false;
static int s_CorrectCount = 0;
static int s_TotalTrails = 0;
static int s_CurrentDataset = 0;
static float s_InputTrainingData[][2] = {
    {0, 0},
    {0, 1},
    {1, 0},
    {1, 1},
};
static float s_ExpectedOutput[] = {
    0, 
    1, 
    1, 
    0
};



static void DisplayStats(const neuralnet *NN, bool IsTraining)
{
    printf("\n"
           "IsTraining:       %s\n", IsTraining? "true" : "false");
    printf("Learning rate:    %f\n", s_LearningRate);
    printf("Current dataset:\n");
    printf("    [%g %g | %g]\n", 
        s_InputTrainingData[s_CurrentDataset][0], 
        s_InputTrainingData[s_CurrentDataset][1], 
        s_ExpectedOutput[s_CurrentDataset]
    );
    printf("Neural network states: \n");
    NeuralNet_Print(NN);
    printf("Correct/Attempts: %d/%d (%g%%)\n", s_CorrectCount, s_TotalTrails, (double)s_CorrectCount / s_TotalTrails * 100);
    printf("Was correct: %s\n\n", s_Correct? "true" : "false");
}

static void DoTraining(neuralnet *NN, bool IsTraining)
{
    neuralnet_training_config TrainingConfig = {
        .LearningRate = s_LearningRate,
        .ExpectedOutputCount = 1,
        .ExpectedOutputs = s_ExpectedOutput,
        .InputCount = 2,
        .Inputs = s_InputTrainingData[s_CurrentDataset],
    };
    if (IsTraining)
    {
        NeuralNet_Train(NN, &TrainingConfig);
    }
    else
    {
        memcpy(NN->Inputs, s_InputTrainingData[s_CurrentDataset], sizeof(s_InputTrainingData[0]));
        NeuralNet_FeedForward(NN, NULL);
    }


    s_TotalTrails++;
    s_Correct = (fabs(NeuralNet_GetOutput(NN)[0] - s_ExpectedOutput[s_CurrentDataset]) < 0.1);
    if (s_Correct)
    {
        s_CorrectCount++;
    }
}

int main(void)
{
    srand(time(NULL));
    int NodeCounts[] = { 2, 1 };
    neuralnet NN = NeuralNet_Create(&(neuralnet_config) {
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
        case 'b':
        {
            IsTraining = !IsTraining;
        } break;
        case 'r':
        {
            DoTraining(&NN, IsTraining);
            DisplayStats(&NN, IsTraining);
        } break;
        case 'a':
        {
            NeuralNet_Randomize(&NN);
        } break;
        case 'R':
        {
            s_TotalTrails = 0;
            s_CorrectCount = 0;
            for (int i = 0; i < 30000; i++)
            {
                s_CurrentDataset = ((float)rand() / RAND_MAX * 4);
                DoTraining(&NN, IsTraining);
                DisplayStats(&NN, IsTraining);
                usleep(10);
            }
        } break;
        default:
        {
            s_CurrentDataset = ((float)rand() / RAND_MAX * 4);
            DoTraining(&NN, IsTraining);
            DisplayStats(&NN, IsTraining);
        } break;
        }
    }

    NeuralNet_Destroy(&NN);
    return 0;
}
