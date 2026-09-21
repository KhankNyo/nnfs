
#include <stdio.h>
#include <assert.h>
#include <stdint.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "extern/stb_image_write.h"

#define TEST_IMAGE_WIDTH 28
#define TEST_IMAGE_HEIGHT 28
#define TEST_IMAGE_CHANNEL_COUNT 4


int main(int ArgumentCount, char **Arguments)
{
    if (ArgumentCount != 2)
    {
        printf("Usage: %s [test image index]\n", Arguments[0]);
        return 1;
    }

    int TestImageIndex = atoi(Arguments[1]);
    const char *OutputFileName = "test.bmp";

    /* terrible code */
    /* https://github.com/phoebetronic/mnist/tree/main */
    FILE *TrainingFile = fopen("mnist_train.csv", "rb");
    assert(TrainingFile);
    {
        for (int i = 0; i < TestImageIndex && !feof(TrainingFile); i++)
        {
            char Ch = 0;
            do {
                Ch = fgetc(TrainingFile);
            } while (Ch && Ch != '\n');
        }

        if (feof(TrainingFile))
        {
            printf("Index too large\n");
            return 1;
        }

        int Digit = 0;
        fscanf(TrainingFile, "%d,", &Digit); /* this sucks */
        printf("Digit: %d\n", Digit);

        static uint8_t ImageData[TEST_IMAGE_WIDTH * TEST_IMAGE_HEIGHT * TEST_IMAGE_CHANNEL_COUNT];
        for (int y = 0; y < TEST_IMAGE_HEIGHT; y++)
        {
            for (int x = 0; x < TEST_IMAGE_WIDTH; x++)
            {
                int Index = y*TEST_IMAGE_WIDTH + x;
                uint8_t *Ptr = ImageData + Index*TEST_IMAGE_CHANNEL_COUNT;

                uint32_t Color = 0;
                fscanf(TrainingFile, "%d,", &Color);
                Color |= Color << 8 | Color << 16 | 0xFF000000;

                memcpy(Ptr, &Color, sizeof Color);
            }
        }

        printf("Writing digit %d to %s\n", Digit, OutputFileName);
        stbi_write_bmp(OutputFileName, TEST_IMAGE_WIDTH, TEST_IMAGE_HEIGHT, TEST_IMAGE_CHANNEL_COUNT, ImageData);
    }
    fclose(TrainingFile);
    return 0;
}

