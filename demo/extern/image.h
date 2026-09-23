#ifndef IMAGE_H
#define IMAGE_H

/* this file is independent of the icommon library and has no dependencies. */


typedef unsigned long long image_u64;
typedef unsigned int image_u32;
typedef unsigned char image_u8;

typedef struct image_resize_config image_resize_config;
typedef struct image_pixel_op image_pixel_op;
typedef image_u64 (*image_pixel_add)(image_u64 A, image_u64 B);
typedef image_u64 (*image_pixel_mul)(image_u64 A, float ValueBetweenZeroAnd1);
typedef image_u64 (*image_pixel_read)(const void *Src, int Bytes);
typedef void (*image_pixel_write)(void *Dst, image_u64 Pixel, int Bytes);

typedef enum 
{
    IMAGE_RESIZE_METHOD_BILINEAR = 0,
    IMAGE_RESIZE_METHOD_NEAREST_NEIGHBOR,
} image_resize_method;

typedef enum
{
    IMAGE_PIXEL_FORMAT_RGBA32 = 0,
    IMAGE_PIXEL_FORMAT_ARGB32,
    IMAGE_PIXEL_FORMAT_BGRA32,
    IMAGE_PIXEL_FORMAT_RGB24,
    IMAGE_PIXEL_FORMAT_BGR24,
    IMAGE_PIXEL_FORMAT_CUSTOM,
} image_pixel_format;

struct image_pixel_op
{
    image_pixel_read Read;
    image_pixel_write Write;
    image_pixel_mul Mul;
    image_pixel_add Add;
};

struct image_resize_config
{
    void *Dst;
    int DstWidth, DstHeight;
    const void *Src;
    int SrcWidth, SrcHeight;

    image_pixel_format Format;
    image_resize_method Method;

    image_pixel_op CustomFormatBilinearOps;
    int CustomFormatPixelSizeBytes;
};


void Image_Resize(const image_resize_config *Config);


#endif /* IMAGE_H */


#if (defined(IMAGE_IMPLEMENTATION) || defined(COMMON_IMPLEMENTATION)) && !defined(IMAGE_ALREADY_IMPLEMENTED)
#define IMAGE_ALREADY_IMPLEMENTED


#if defined(IMAGE_DONT_USE_MEMCPY)
static void Image__Memcpy(void *Dst, const void *Src, int SizeBytes)
{
    image_u8 *DstPtr = Dst;
    image_u8 *SrcPtr = Src;
    while (SizeBytes --> 0)
        *DstPtr++ = *SrcPtr++;
}

#else

#include <string.h>

static void Image__Memcpy(void *Dst, const void *Src, int SizeBytes)
{
    memcpy(Dst, Src, SizeBytes);
}
#endif



static void Image__ResizeLinear(void *Dst, const void *Src, int Width, int Height, int NewWidth, int NewHeight, int PixelSize)
{
    image_u8 *DstPtr = Dst;
    const image_u8 *SrcPtr = Src;
    float WScale = (float)Width / NewWidth;
    float HScale = (float)Height / NewHeight;
    for (int y = 0; y < NewHeight; y++)
    {
        for (int x = 0; x < NewWidth; x++)
        {
            int OldX = x * WScale;
            int OldY = y * HScale;
            int Index = (OldX + OldY * Width) * PixelSize;

            Image__Memcpy(DstPtr, SrcPtr + Index, PixelSize);
            DstPtr += PixelSize;
        }
    }
}

static void Image__ResizeBilinearArbitrary(
    image_u8 *Dst, 
    const image_u8 *Src, 
    int Width, int Height, 
    int NewWidth, int NewHeight, 
    int PixelSizeBytes,
    const image_pixel_op *Op
) {
    float WScale = (float)(Width - 1) / (NewWidth);
    float HScale = (float)(Height - 1) / (NewHeight);
    for (int y = 0; y < NewHeight; y++)
    {
        for (int x = 0; x < NewWidth; x++)
        {
            int OldX0 = x * WScale;
            int OldY0 = y * HScale;
            int OldX1 = x * WScale + 1;
            int OldY1 = y * HScale + 1;
            float WeightX = x*WScale - (float)OldX0;
            float WeightY = y*HScale - (float)OldY0;
            int IndexA = (OldX0 + OldY0 * Width);
            int IndexB = (OldX1 + OldY0 * Width);
            int IndexC = (OldX0 + OldY1 * Width);
            int IndexD = (OldX1 + OldY1 * Width);

            image_u64 A = Op->Read(Src + IndexA*PixelSizeBytes, PixelSizeBytes);
            image_u64 B = Op->Read(Src + IndexB*PixelSizeBytes, PixelSizeBytes);
            image_u64 C = Op->Read(Src + IndexC*PixelSizeBytes, PixelSizeBytes);
            image_u64 D = Op->Read(Src + IndexD*PixelSizeBytes, PixelSizeBytes);
            A = Op->Mul(A, (1 - WeightX) * (1 - WeightY));
            B = Op->Mul(B, WeightX * (1 - WeightY));
            C = Op->Mul(C, (1 - WeightX) * WeightY);
            D = Op->Mul(D, WeightX * WeightY);
            image_u64 Result = Op->Add(A, Op->Add(B, Op->Add(C, D)));
            Op->Write(Dst, Result, PixelSizeBytes);
            Dst += PixelSizeBytes;
        }
    }
}

static int Image__GetPixelSizeBytes(const image_resize_config *Config)
{
    switch (Config->Format)
    {
    case IMAGE_PIXEL_FORMAT_RGBA32:
    case IMAGE_PIXEL_FORMAT_ARGB32:
    case IMAGE_PIXEL_FORMAT_BGRA32:
        return 4;
    case IMAGE_PIXEL_FORMAT_RGB24:
    case IMAGE_PIXEL_FORMAT_BGR24:
        return 3;
    case IMAGE_PIXEL_FORMAT_CUSTOM: 
        return Config->CustomFormatPixelSizeBytes;
    }
    return 0;
}



static image_u64 Image__ReadBytes(const void *Src, int PixelCount) 
{
    image_u64 Result = 0;
    const image_u8 *SrcPtr = Src;
    for (int i = 0; i < PixelCount; i++)
        Result |= (image_u64)SrcPtr[i] << 8*i;
    return Result;
}


static void Image__WriteBytes(void *Dst, image_u64 Src, int PixelCount)
{
    image_u8 *DstPtr = Dst;
    for (int i = 0; i < PixelCount; i++)
        DstPtr[i] = Src >> i*8;
}

static image_u64 Image__AddBytes(image_u64 A, image_u64 B)
{
    /* NOTE: adding 8-bit lanes, each lanes are independent, wrap on unsigned overflow */
    image_u64 Mask = 0x8080808080808080;
    image_u64 Sum = (A & ~Mask) + (B & ~Mask);
    image_u64 Carry = (A ^ B) & Mask;
    return Sum ^ Carry;
}

static image_u64 Image__Mul4Bytes(image_u64 A, float Value)
{
    image_u32 Color = A;
    image_u32 Result = 0;
    for (int i = 0; i < 4; i++)
    {
        image_u8 Channel = Color >> i*8;
        Result |= (image_u32)(image_u8)(Channel * Value) << i*8;
    }
    return Result;
}

static image_u64 Image__MulAXXX32(image_u64 Color, float Value)
{
    image_u8 A = Color >> 24;
    image_u8 B = (((Color >> 16) & 0xFF) * Value);
    image_u8 G = (((Color >> 8) & 0xFF) * Value);
    image_u8 R = (((Color >> 0) & 0xFF) * Value);
    image_u32 Result = 
        (image_u32)A << 24
        | (image_u32)B << 16
        | (image_u32)G << 8
        | (image_u32)R << 0;
    return Result;
}

static image_u64 Image__MulXXXA32(image_u64 Color, float Value)
{
    image_u8 A = Color & 0xFF;
    image_u8 B = (((Color >> 8) & 0xFF) * Value);
    image_u8 G = (((Color >> 16) & 0xFF) * Value);
    image_u8 R = (((Color >> 24) & 0xFF) * Value);
    image_u32 Result = 
        (image_u32)R << 24
        | (image_u32)G << 16
        | (image_u32)B << 8
        | (image_u32)A << 0;
    return Result;
}


static image_pixel_op Image__GetPixelOp(image_pixel_format Format)
{
    image_pixel_op Result = { 0 };
    Result.Read = Image__ReadBytes;
    Result.Write = Image__WriteBytes;
    Result.Add = Image__AddBytes;
    Result.Mul = Image__Mul4Bytes;
    switch (Format)
    {
    case IMAGE_PIXEL_FORMAT_BGRA32:
    case IMAGE_PIXEL_FORMAT_RGBA32:
        Result.Add = Image__AddBytes;
        Result.Mul = Image__MulXXXA32;
        break;
    case IMAGE_PIXEL_FORMAT_ARGB32:
        Result.Add = Image__AddBytes;
        Result.Mul = Image__MulAXXX32;
        break;
    case IMAGE_PIXEL_FORMAT_BGR24:
        Result.Add = Image__AddBytes;
        Result.Mul = Image__Mul4Bytes;
        break;
    case IMAGE_PIXEL_FORMAT_RGB24:
        Result.Add = Image__AddBytes;
        Result.Mul = Image__Mul4Bytes;
        break;
    default: break;
    }
    return Result;
}

void Image_Resize(const image_resize_config *Config)
{
    int PixelSizeBytes = Image__GetPixelSizeBytes(Config);
    switch (Config->Method)
    {
    case IMAGE_RESIZE_METHOD_NEAREST_NEIGHBOR:
    {
        Image__ResizeLinear(
            Config->Dst, 
            Config->Src, 
            Config->SrcWidth, Config->SrcHeight, 
            Config->DstWidth, Config->DstHeight,
            PixelSizeBytes
        );
    } break;
    case IMAGE_RESIZE_METHOD_BILINEAR:
    {
        image_pixel_op Ops = Image__GetPixelOp(Config->Format);
        if (Config->CustomFormatBilinearOps.Add) Ops.Add = Config->CustomFormatBilinearOps.Add;
        if (Config->CustomFormatBilinearOps.Mul) Ops.Mul = Config->CustomFormatBilinearOps.Mul;
        if (Config->CustomFormatBilinearOps.Read) Ops.Read = Config->CustomFormatBilinearOps.Read;
        if (Config->CustomFormatBilinearOps.Write) Ops.Write = Config->CustomFormatBilinearOps.Write;

        Image__ResizeBilinearArbitrary(
            Config->Dst, 
            Config->Src, 
            Config->SrcWidth, Config->SrcHeight, 
            Config->DstWidth, Config->DstHeight, 
            PixelSizeBytes,
            &Ops
        );
    } break;
    }
}


#endif
