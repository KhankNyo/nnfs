#ifndef NND_H
#define NND_H

#include <stdint.h>


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
    nnd_type WeightType;      /* type for compressed weight */
    nnd_type QScalarInvType;  /* type for quantization scalar factor (inverse, preferably machine word type) */
    const char *FilePathNoExtension;
    int QScalarFxpShift;                    /* output QScalarInv = 2^(QScalarFxpShift + QScalarInvShift) / (QScalar * 2^QScalarFxpShift) */

    union nnd_file_format_config {
        struct {
            nnd_type InfoType;        /* type for integer info (layer count, input count, preferably integer) */
            const char *HeaderGuard;
            const char *VariableName;
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
    int32_t InputCount, 
    const uint64_t *NodeCount, int32_t LayerCount,
    const float *Weights, int32_t WeightCount,
    nnd_serialize_config *Config, 
    float QScalarInvShifted, int32_t FxpShift
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

static void Nnd__QuantizeValue(nnd_type DstType, void *Dst, float QScalar, float QValue)
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


const char *Nnd__SerializeFp32ToCSource(
    int32_t InputCount, 
    const uint64_t *NodeCount, int32_t LayerCount,
    const float *Weights, int32_t WeightCount,
    nnd_serialize_config *Config, 
    float QScalarInvShifted, int32_t FxpShift, float QScalar
) {
    const nnd_file_format_config *FConfig = &Config->FileFormatConfig;
    const char *WeightType = Nnd__NumericTypeToCType(Config->WeightType);
    const char *QScalarInvType = Nnd__NumericTypeToCType(Config->QScalarInvType);
    const char *IType = Nnd__NumericTypeToCType(FConfig->C.InfoType);

    char HeaderFileName[256], SourceFileName[256];
    snprintf(HeaderFileName, sizeof HeaderFileName, "%s.h", Config->FilePathNoExtension);
    snprintf(SourceFileName, sizeof SourceFileName, "%s.c", Config->FilePathNoExtension);

    const char *VariableName = FConfig->C.VariableName;

    {
        FILE *HeaderFile = fopen(HeaderFileName, "wb");
        if (HeaderFile) 
        {
            fprintf(HeaderFile, "/* Generated by neuralnet.h */\n");
            fprintf(HeaderFile, "#ifndef %s\n", FConfig->C.HeaderGuard);
            fprintf(HeaderFile, "#define %s\n", FConfig->C.HeaderGuard);
            fprintf(HeaderFile, "\n#include <stdint.h>\n");
            fprintf(HeaderFile, "\nextern const ");
            NND__PRINT_STRUCT_DEF(HeaderFile, IType, QScalarInvType, WeightType);
            fprintf(HeaderFile, " %s;", VariableName);
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

            fprintf(SourceFile, "\nconst %s s_NodeCountPerLayer[] = {", IType);
            for (int i = 0; i < LayerCount; i++)
            {
                Nnd__SerializeValueToCSource(SourceFile, FConfig->C.InfoType, NND_TYPE_INT32, &NodeCount[i]);
                fprintf(SourceFile, ", ");
            }
            fprintf(SourceFile, "};");

            fprintf(SourceFile, "\nconst %s s_Weights[] = {", WeightType);
            for (int64_t i = 0; i < WeightCount; i++)
            {
                char QWeight[8] = { 0 };
                Nnd__QuantizeValue(Config->WeightType, QWeight, QScalar, Weights[i]);
                Nnd__SerializeValueToCSource(SourceFile, Config->WeightType, Config->WeightType, QWeight);
                fprintf(SourceFile, ", ");
            }
            fprintf(SourceFile, "};");


            fprintf(SourceFile, "\nconst typeof(%s) %s = {", VariableName, VariableName);
            fprintf(SourceFile, "\n    .InputCount = ");
            Nnd__SerializeValueToCSource(SourceFile, FConfig->C.InfoType, NND_TYPE_INT32, &InputCount);
            fprintf(SourceFile, ",\n    .LayerCount = ");
            Nnd__SerializeValueToCSource(SourceFile, FConfig->C.InfoType, NND_TYPE_INT32, &LayerCount);
            fprintf(SourceFile, ",\n    .QScalarInvShifted = ");
            Nnd__SerializeValueToCSource(SourceFile, FConfig->C.InfoType, NND_TYPE_FP32, &QScalarInvShifted);
            fprintf(SourceFile, ",\n    .QScalarInvShamt = ");
            Nnd__SerializeValueToCSource(SourceFile, FConfig->C.InfoType, NND_TYPE_INT32, &FxpShift);
            fprintf(SourceFile, ",\n    .NodeCountPerLayer = s_NodeCountPerLayer");
            fprintf(SourceFile, ",\n    .Weights = s_Weights");
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

const char *Nnd__SerializeFp32ToNnd(
    int32_t InputCount, 
    const uint64_t *NodeCount, int32_t LayerCount,
    const float *Weights, int32_t WeightCount,
    nnd_serialize_config *Config, 
    float QScalarInvShifted, int32_t FxpShift, float QScalar
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
    nnd_type WType = Config->WeightType;
    nnd_type QType = Config->QScalarInvType;

    int Ao = sizeof(nnd_file_header) + LayerCount*8;
    int Wo = Ao;

    nnd_file_header Header = {
        .Magic = NND_FILE_MAGIC,
        .Version = NND_FILE_VERSION,
        .Flags = Config->Flags,
        .Ao = Ao,
        .As = 0,
        .Wo = Wo,
        .Wsb = WeightCount*Nnd_GetTypeSize(WType),
        .Wt = WType,
        .WDecimalPlace = 0,
        .QScalarType = QType,
        .QScalarShamt = FxpShift,
        .QScalarInvShifted = { 0 },
        .Lc = LayerCount,
        .Ic = InputCount,
    };
    Nnd__SerializeValueToMemory(Header.QScalarInvShifted, QType, NND_TYPE_FP32, &QScalarInvShifted);
    fwrite(&Header, 1, sizeof Header, f);
    fwrite(NodeCount, sizeof(NodeCount[0]), LayerCount, f);
    int QuantizedWeightSize = Nnd_GetTypeSize(WType);
    for (int i = 0; i < WeightCount; i++)
    {
        uint8_t Value[8] = { 0 };
        Nnd__QuantizeValue(WType, Value, QScalar, Weights[i]);
        fwrite(Value, QuantizedWeightSize, QuantizedWeightSize, f);
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
    int32_t InputCount, 
    const uint64_t *NodeCount, int32_t LayerCount,
    const float *Weights, int32_t WeightCount,
    nnd_serialize_config *Config, 
    float QScalarInvShifted, int32_t FxpShift
) {
    const char *ErrorMessage = NULL;

    float QScalar = (float)(1ll << FxpShift) / QScalarInvShifted;
    switch (Config->FileFormat)
    {
    case NND_FILE_FORMAT_C:
    {
        ErrorMessage = Nnd__SerializeFp32ToCSource(
            InputCount,
            NodeCount, LayerCount, 
            Weights, WeightCount,
            Config, 
            QScalarInvShifted, FxpShift, QScalar
        );
    } break;
    case NND_FILE_FORMAT_NND:
    {
        ErrorMessage = Nnd__SerializeFp32ToNnd(
            InputCount, 
            NodeCount, LayerCount,
            Weights, WeightCount,
            Config, 
            QScalarInvShifted, FxpShift, QScalar
        );
    } break;
    }
    return ErrorMessage;
}

#endif



