#ifndef NND_H
#define NND_H

#include <stdint.h>
#include "neuralnet.h"
#include "nnfxp.h"


#define NND_FILE_MAGIC {'n','n','d','f'}
#define NND_FILE_VERSION 1

typedef struct nnd_serialize_config nnd_serialize_config;
typedef struct nnd_file_header nnd_file_header;
typedef union nnd_file_format_config nnd_file_format_config;
typedef enum 
{
    NND_FILE_FORMAT_C = 0,
    NND_FILE_FORMAT_NND = 1,
} nnd_file_format;
typedef enum
{
    NND_TYPE_INT8 = 0,
    NND_TYPE_INT16 = 1,
    NND_TYPE_INT32 = 2,
    NND_TYPE_FP32 = 3,
} nnd_type;
typedef enum
{
    NND_FLAG_NONE = 0,
    NND_FLAG_ENABLE_QUANTIZATION = 1 << 0,
    NND_FLAG_SPARSE_WEIGHTS = 1 << 1,
} nnd_serialize_flags;

struct nnd_serialize_config
{
    nnd_serialize_flags Flags;
    nnd_file_format FileFormat;
    const char *FilePathNoExtension;
    nnd_type QWeightType;   /* type for quantized weight */
    nnd_type DQWeightType;  /* type for dequantized weight (same as quantization scalar type) */
    int DQWeightFxpDecimal; /* if dequantized weights are fixed point, the value is the decimal place of those fixed point weights 
                               otherwise should be 0 */
    union nnd_file_format_config {
        struct {
            nnd_type InfoType;        /* type for integer info (layer count, input count, preferably integer) */
            const char *HeaderGuard;
            const char *ModelName;
        } C;
        struct {
        } Nnd;
    } FileFormatConfig;
};

struct nnd_file_header
{
    char Magic[4];
    uint32_t Version;
    uint64_t Flags;
    uint32_t Ao;
    uint32_t As;
    uint64_t Wo;
    uint64_t Wsb;
    uint32_t Wt;
    uint32_t WDecimalPlace;
    uint32_t QScalarType;
    uint32_t QScalarShamt;
    uint8_t QScalarInvShifted[8];
    uint32_t Lc;
    uint32_t Ic;
};

const char *Nnd_SerializeFp32(
    neuralnet *NN, 
    nnd_serialize_config *Config
);
int Nnd_GetTypeSize(nnd_type Type); /* returns size in bytes */
float Nnd_GetTypeMax(nnd_type Type);


#endif /* NND_H */


#if defined(NND_IMPLEMENTATION) && !defined(NND_ALREADY_IMPLEMENTED)
#define NND_ALREADY_IMPLEMENTED

#define NND__MAX(a, b) ((a) > (b)? (a) : (b))
#define NND__MIN(a, b) ((a) < (b)? (a) : (b))
#define NND__IN_RANGE(lower, n, upper) ((lower) <= (n) && (n) <= (upper))
#define NND__CLAMP(lower, n, upper) NND__MIN(NND__MAX(lower, n), upper)
#define NND__PRINT_STRUCT_DEF(f, itype, qtype, wtype) fprintf(f, \
        "struct {\n"\
        "   %s InputCount;\n"\
        "   %s LayerCount;\n"\
        "   %s QScalarInvShifted;\n"\
        "   %s QScalarInvShamt;\n"\
        "   const %s *NodeCountPerLayer;\n"\
        "   const %s *Weights;\n"\
        "} ", itype, itype, qtype, itype, itype, wtype\
        )

#include <stdio.h>
#include <assert.h>
#include <stdbool.h>
#include <string.h>
#include <float.h>


static int32_t Nnd__GetInt32(nnd_type SrcType, const void *Src)
{
    int32_t Result = 0;
    switch (SrcType)
    {
    case NND_TYPE_INT8:
    {
        int8_t Tmp = 0;
        memcpy(&Tmp, Src, 1);
        Result = Tmp;
    } break;
    case NND_TYPE_INT16:
    {
        int16_t Tmp = 0;
        memcpy(&Tmp, Src, 2);
        Result = Tmp;
    } break;
    case NND_TYPE_INT32:
    {
        int32_t Tmp = 0;
        memcpy(&Tmp, Src, 4);
        Result = Tmp;
    } break;
    case NND_TYPE_FP32:
    {
        float Tmp = 0;
        memcpy(&Tmp, Src, 4);
        Result = Tmp;
    } break;
    }
    return Result;
}

static float Nnd__GetFp32(nnd_type SrcType, const void *Src)
{
    float Result = 0;
    switch (SrcType)
    {
    case NND_TYPE_INT8:
    {
        int8_t Tmp = 0;
        memcpy(&Tmp, Src, 1);
        Result = Tmp;
    } break;
    case NND_TYPE_INT16:
    {
        int16_t Tmp = 0;
        memcpy(&Tmp, Src, 2);
        Result = Tmp;
    } break;
    case NND_TYPE_INT32:
    {
        int32_t Tmp = 0;
        memcpy(&Tmp, Src, 4);
        Result = Tmp;
    } break;
    case NND_TYPE_FP32:
    {
        float Tmp = 0;
        memcpy(&Tmp, Src, 4);
        Result = NND__CLAMP(INT32_MIN, Tmp, INT32_MAX);
    } break;
    }
    return Result;
}

static void Nnd__SerializeValueToCSource(FILE *File, nnd_type DstType, nnd_type SrcType, const void *Src)
{
    switch (DstType)
    {
    case NND_TYPE_INT8:
    {
        int32_t Value = Nnd__GetInt32(SrcType, Src);
        Value = NND__CLAMP(INT8_MIN, Value, INT8_MAX);
        fprintf(File, "%d", Value);
    } break;
    case NND_TYPE_INT16:
    {
        int32_t Value = Nnd__GetInt32(SrcType, Src);
        Value = NND__CLAMP(INT16_MIN, Value, INT16_MAX);
        fprintf(File, "%d", Value);
    } break;
    case NND_TYPE_INT32:
    {
        int32_t Value = Nnd__GetInt32(SrcType, Src);
        fprintf(File, "%d", Value);
    } break;
    case NND_TYPE_FP32:
    {
        float Value = Nnd__GetFp32(SrcType, Src);
        fprintf(File, "%f", Value);
    } break;
    }
}

static void Nnd__SerializeValueToMemory(uint8_t Dst[8], nnd_type DstType, nnd_type SrcType, const void *Src)
{
    switch (DstType)
    {
    case NND_TYPE_INT8:
    {
        int32_t Value = Nnd__GetInt32(SrcType, Src);
        Value = NND__CLAMP(INT8_MIN, Value, INT8_MAX);
        memcpy(Dst, &Value, sizeof Value);
    } break;
    case NND_TYPE_INT16:
    {
        int32_t Value = Nnd__GetInt32(SrcType, Src);
        Value = NND__CLAMP(INT16_MIN, Value, INT16_MAX);
        memcpy(Dst, &Value, sizeof Value);
    } break;
    case NND_TYPE_INT32:
    {
        int32_t Value = Nnd__GetInt32(SrcType, Src);
        memcpy(Dst, &Value, sizeof Value);
    } break;
    case NND_TYPE_FP32:
    {
        float Value = Nnd__GetFp32(SrcType, Src);
        memcpy(Dst, &Value, sizeof Value);
    } break;
    }
}

static void Nnd__QuantizeValue(void *Dst, nnd_type DstType, float QScalar, float QValue)
{
    float Result = QScalar * QValue + 0.5;
    switch (DstType)
    {
    case NND_TYPE_FP32:
    {
        memcpy(Dst, &Result, sizeof Result);
    } break;
    case NND_TYPE_INT8:
    {
        int8_t Value = NND__CLAMP(INT8_MIN, Result, INT8_MAX);
        memcpy(Dst, &Value, sizeof Value);
    } break;
    case NND_TYPE_INT16:
    {
        int16_t Value = NND__CLAMP(INT16_MIN, Result, INT16_MAX);
        memcpy(Dst, &Value, sizeof Value);
    } break;
    case NND_TYPE_INT32:
    {
        int32_t Value = NND__CLAMP(INT32_MIN, Result, INT32_MAX);
        memcpy(Dst, &Value, sizeof Value);
    } break;
    }
}

static const char *Nnd__NumericTypeToCType(nnd_type NumericType)
{
    const char *Type = NULL;
    switch (NumericType)
    {
    case NND_TYPE_FP32: Type = "float"; break;
    case NND_TYPE_INT8: Type = "int8_t"; break;
    case NND_TYPE_INT16: Type = "int16_t"; break;
    case NND_TYPE_INT32: Type = "int32_t"; break;
    }
    return Type;
}

static void Nnd__FWriteLE(FILE *Dst, const void *Value, nnd_type Type)
{
    uint32_t IntRep = 0;
    int TypeSize = Nnd_GetTypeSize(Type);
    assert(TypeSize < (int)sizeof(IntRep));

    memcpy(&IntRep, Value, TypeSize);
    for (int i = 0; i < TypeSize; i++)
    {
        uint8_t Byte = IntRep >> (i*8);
        fwrite(&Byte, 1, 1, Dst);
    }
}


static const char *Nnd__SerializeFp32ToCSource(
    neuralnet *NN,
    nnd_serialize_config *Config, 
    float Scalar, float ScalarInvShifted, int32_t ScalarShift
) {
    const nnd_file_format_config *FConfig = &Config->FileFormatConfig;
    const char *QWeightType = Nnd__NumericTypeToCType(Config->QWeightType);
    const char *DQWeightType = Nnd__NumericTypeToCType(Config->DQWeightType);
    const char *InfoType = Nnd__NumericTypeToCType(FConfig->C.InfoType);

    char HeaderFileName[256], SourceFileName[256];
    snprintf(HeaderFileName, sizeof HeaderFileName, "%s.h", Config->FilePathNoExtension);
    snprintf(SourceFileName, sizeof SourceFileName, "%s.c", Config->FilePathNoExtension);

    const char *ModelName = FConfig->C.ModelName;

    {
        FILE *HeaderFile = fopen(HeaderFileName, "wb");
        if (HeaderFile) 
        {
            fprintf(HeaderFile, "/* Generated by neuralnet.h */\n");
            fprintf(HeaderFile, "#ifndef %s\n", FConfig->C.HeaderGuard);
            fprintf(HeaderFile, "#define %s\n", FConfig->C.HeaderGuard);
            fprintf(HeaderFile, "\n#include <stdint.h>\n");
            fprintf(HeaderFile, "\nextern const ");
            NND__PRINT_STRUCT_DEF(HeaderFile, InfoType, DQWeightType, QWeightType);
            fprintf(HeaderFile, " %s;", ModelName);
            fprintf(HeaderFile, "\n#endif /* %s */", FConfig->C.HeaderGuard);
        }
        else
        {
            return "Unable to open header file";
        }
        fclose(HeaderFile);
    }

    {
        FILE *SourceFile = fopen(SourceFileName, "wb");
        if (SourceFile)
        {
            fprintf(SourceFile, "/* Generated by neuralnet.h */\n");
            fprintf(SourceFile, "\n#include \"%s\"\n", HeaderFileName);

            fprintf(SourceFile, "\nconst %s s_NodeCountPerLayer[] = {", InfoType);
            for (int i = 0; i < NN->LayerCount; i++)
            {
                static_assert(sizeof(NN->Layers[i].OutputCount) == sizeof(int32_t), "");
                Nnd__SerializeValueToCSource(SourceFile, FConfig->C.InfoType, NND_TYPE_INT32, &NN->Layers[i].OutputCount);
                fprintf(SourceFile, ", ");
            }
            fprintf(SourceFile, "};");

            fprintf(SourceFile, "\nconst %s s_QWeights[] = {", QWeightType);
            for (int64_t i = 0; i < NN->WeightCount; i++)
            {
                uint8_t QWeight[8] = { 0 };
                Nnd__QuantizeValue(QWeight, Config->QWeightType, Scalar, NN->Weights[i]);
                Nnd__SerializeValueToCSource(SourceFile, Config->QWeightType, Config->QWeightType, QWeight);
                fprintf(SourceFile, ", ");
            }
            fprintf(SourceFile, "};");


            fprintf(SourceFile, "\nconst typeof(%s) %s = {", ModelName, ModelName);
            fprintf(SourceFile, "\n    .InputCount = ");
            Nnd__SerializeValueToCSource(SourceFile, FConfig->C.InfoType, NND_TYPE_INT32, &NN->InputCount);
            fprintf(SourceFile, ",\n    .LayerCount = ");
            Nnd__SerializeValueToCSource(SourceFile, FConfig->C.InfoType, NND_TYPE_INT32, &NN->LayerCount);
            fprintf(SourceFile, ",\n    .QScalarInvShifted = ");
            Nnd__SerializeValueToCSource(SourceFile, FConfig->C.InfoType, NND_TYPE_FP32, &ScalarInvShifted);
            fprintf(SourceFile, ",\n    .QScalarInvShamt = ");
            Nnd__SerializeValueToCSource(SourceFile, FConfig->C.InfoType, NND_TYPE_INT32, &ScalarShift);
            fprintf(SourceFile, ",\n    .NodeCountPerLayer = s_NodeCountPerLayer");
            fprintf(SourceFile, ",\n    .Weights = s_QWeights");
            fprintf(SourceFile, "};\n");
        }
        else
        {
            return "Unable to open source file";
        }
        fclose(SourceFile);
    }

    return NULL;
}

static const char *Nnd__SerializeFp32ToNnd(
    neuralnet *NN,
    nnd_serialize_config *Config,
    float Scalar, float ScalarInvShifted, int32_t ScalarShift
) {
    char FileName[256];
    snprintf(FileName, sizeof FileName, "%s.nnd", Config->FilePathNoExtension);

    FILE *f = fopen(FileName, "wb");
    if (!f)
    {
        return "Unable to open file for writing";
    }

    static_assert(sizeof(nnd_file_header) == 72, "");
    assert(!(Config->Flags & NND_FLAG_SPARSE_WEIGHTS) && "TODO: sparse weights");
    nnd_type QWeightType = Config->QWeightType;
    nnd_type DQWeightType = Config->DQWeightType;

    int Ao = sizeof(nnd_file_header) + NN->LayerCount*8;
    int Wo = Ao;

    /* header */
    {
        nnd_file_header Header = {
            .Magic = NND_FILE_MAGIC,
            .Version = NND_FILE_VERSION,
            .Flags = Config->Flags,
            .Ao = Ao,
            .As = 0,
            .Wo = Wo,
            .Wsb = NN->WeightCount*Nnd_GetTypeSize(QWeightType),
            .Wt = QWeightType,
            .WDecimalPlace = 0,
            .QScalarType = DQWeightType,
            .QScalarShamt = ScalarShift,
            .QScalarInvShifted = { 0 },
            .Lc = NN->LayerCount,
            .Ic = NN->InputCount,
        };
        Nnd__SerializeValueToMemory(Header.QScalarInvShifted, DQWeightType, NND_TYPE_FP32, &ScalarInvShifted);
        fwrite(&Header, 1, sizeof Header, f);

        /* node count per layer */
        for (int i = 0; NN->LayerCount; i++)
        {
            static_assert(sizeof(NN->Layers[i].OutputCount) == sizeof(int32_t), "");
            Nnd__FWriteLE(f, &NN->Layers[i].OutputCount, NND_TYPE_INT32);
        }
    }

    for (int i = 0; i < NN->WeightCount; i++)
    {
        uint8_t Value[8] = { 0 };
        Nnd__QuantizeValue(Value, QWeightType, Scalar, NN->Weights[i]);
        Nnd__FWriteLE(f, Value, QWeightType);
    }

    fclose(f);
    return NULL;
}

/* returns size in bytes */
int Nnd_GetTypeSize(nnd_type Type)
{
    switch (Type)
    {
    case NND_TYPE_INT32: return 4;
    case NND_TYPE_INT16: return 2;
    case NND_TYPE_INT8: return 1;
    case NND_TYPE_FP32: return 4;
    }
    return 0;
}

float Nnd_GetTypeMax(nnd_type Type)
{
    switch (Type)
    {
    case NND_TYPE_INT32: return INT32_MAX;
    case NND_TYPE_INT16: return INT16_MAX;
    case NND_TYPE_INT8: return INT8_MAX;
    case NND_TYPE_FP32: return FLT_MAX;
    }
    return 0;
}

const char *Nnd_SerializeFp32(
    neuralnet *NN, 
    nnd_serialize_config *Config
) {
    const char *ErrorMessage = NULL;
    float Scalar = 0;
    float ScalarInvShifted = 0;
    int ScalarShift = 0;
    {
        float QWeightMax = Nnd_GetTypeMax(Config->QWeightType);
        float DQWeightMax = Nnd_GetTypeMax(Config->DQWeightType);

        Scalar = NeuralNet_GetQuantizationScalar(NN, QWeightMax);
        ScalarShift = (int)log2f(Scalar * (DQWeightMax / QWeightMax));  /* b in the equation (dqweight = qweight * s_inv / 2^b) */
        ScalarInvShifted = (float)(1ll << ScalarShift) / Scalar;        /* s_inv = (2^b) / s */
    }

    switch (Config->FileFormat)
    {
    case NND_FILE_FORMAT_C:
    {
        ErrorMessage = Nnd__SerializeFp32ToCSource(
            NN, Config, 
            Scalar, ScalarInvShifted, ScalarShift
        );
    } break;
    case NND_FILE_FORMAT_NND:
    {
        ErrorMessage = Nnd__SerializeFp32ToNnd(
            NN, Config, 
            Scalar, ScalarInvShifted, ScalarShift
        );
    } break;
    }
    return ErrorMessage;
}

#endif



