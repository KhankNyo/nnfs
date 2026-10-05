#ifndef NND_H
#define NND_H

#include <stdint.h>

typedef struct nnd_serialize_config nnd_serialize_config;
typedef struct nnd_file_header nnd_file_header;
typedef enum 
{
    NND_FILE_FORMAT_C = 0,
    NND_FILE_FORMAT_NND = 1,
} nnd_file_format;
typedef enum
{
    NND_TYPE_INT32 = 0,
    NND_TYPE_INT16 = 1,
    NND_TYPE_INT8 = 2,
    NND_TYPE_FP32 = 3,
} nnd_type;
typedef enum
{
    NND_FLAG_ENABLE_QUANTIZATION = 1 << 0,
} nnd_serialize_flags;

struct nnd_serialize_config
{
    nnd_serialize_flags Flags;
    nnd_file_format FileFormat;
    nnd_type WeightType;      /* type for compressed weight */
    nnd_type QScalarInvType;  /* type for quantization scalar factor (inverse, preferably machine word type) */
    nnd_type InfoType;        /* type for integer info (layer count, input count, preferably integer) */
    const char *FilePathNoExtension;
    int QScalarFxpShift;                    /* output QScalarInv = 2^(QScalarFxpShift + QScalarInvShift) / (QScalar * 2^QScalarFxpShift) */

    union {
        struct {
            const char *HeaderGuard;
            const char *VariableName;
        } FileFormatC;
    };
};

struct nnd_file_header
{
    char Magic[4];
    uint32_t Version;
    uint64_t Flags;
    uint64_t Ao;
    uint64_t Wo;
    uint64_t Wsb;
    uint32_t Wt;
    uint32_t WDecimalPlace;
    uint32_t QScalarType;
    uint32_t QScalarShamt;
    uint8_t QScalarInvShifted[8];
    uint64_t Lc;
    uint64_t Ic;
    uint64_t *Nc;
    uint64_t *Wol;
    void *Wd;
};

const char *Nnd_SerializeFp32(
    int32_t InputCount, 
    const int *NodeCount, int32_t LayerCount,
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



static void Nnd__SerializeInt(FILE *File, nnd_type DstType, int32_t Value)
{
    switch (DstType)
    {
    case NND_TYPE_INT8:
    {
        assert(NND__IN_RANGE(INT8_MIN, Value, INT8_MAX));
        fprintf(File, "%d", Value);
    } break;
    case NND_TYPE_INT16:
    {
        assert(NND__IN_RANGE(INT16_MIN, Value, INT16_MAX));
        fprintf(File, "%d", Value);
    } break;
    case NND_TYPE_INT32:
    {
        fprintf(File, "%d", Value);
    } break;
    case NND_TYPE_FP32:
    {
        assert(false && "Unreachable");
    } break;
    }
}

static void Nnd__SerializeValueToCSource(FILE *File, nnd_type DstType, nnd_type SrcType, const void *Src)
{
    switch (SrcType)
    {
    case NND_TYPE_INT8:
    {
        int8_t Value = 0;
        memcpy(&Value, Src, sizeof Value);
        Nnd__SerializeInt(File, DstType, Value);
    } break;
    case NND_TYPE_INT16:
    {
        int16_t Value = 0;
        memcpy(&Value, Src, sizeof Value);
        Nnd__SerializeInt(File, DstType, Value);
    } break;
    case NND_TYPE_INT32:
    {
        int32_t Value = 0;
        memcpy(&Value, Src, sizeof Value);
        Nnd__SerializeInt(File, DstType, Value);
    } break;
    case NND_TYPE_FP32:
    {
        float Value = 0;
        memcpy(&Value, Src, sizeof Value);
        switch (DstType)
        {
        case NND_TYPE_FP32:
        {
            fprintf(File, "%f", Value);
        } break;
        case NND_TYPE_INT8:
        {
            int8_t Result = NND__CLAMP(INT8_MIN, Value, INT8_MAX);
            fprintf(File, "%d", Result);
        } break;
        case NND_TYPE_INT16:
        {
            int16_t Result = NND__CLAMP(INT16_MIN, Value, INT16_MAX);
            fprintf(File, "%d", Result);
        } break;
        case NND_TYPE_INT32:
        {
            int32_t Result = NND__CLAMP(INT32_MIN, Value, INT32_MAX);
            fprintf(File, "%d", Result);
        } break;
        }
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
    const int *NodeCount, int32_t LayerCount,
    const float *Weights, int32_t WeightCount,
    nnd_serialize_config *Config, 
    float QScalarInvShifted, int32_t FxpShift
) {
    const char *WeightType = Nnd__NumericTypeToCType(Config->WeightType);
    const char *QScalarInvType = Nnd__NumericTypeToCType(Config->QScalarInvType);
    const char *IType = Nnd__NumericTypeToCType(Config->InfoType);

    char HeaderFileName[256], SourceFileName[256];
    snprintf(HeaderFileName, sizeof HeaderFileName, "%s.h", Config->FilePathNoExtension);
    snprintf(SourceFileName, sizeof SourceFileName, "%s.c", Config->FilePathNoExtension);

    float QScalar = (float)(1ll << FxpShift) / QScalarInvShifted;
    const char *VariableName = Config->FileFormatC.VariableName;

    {
        FILE *HeaderFile = fopen(HeaderFileName, "wb");
        if (HeaderFile) 
        {
            fprintf(HeaderFile, "/* Generated by neuralnet.h */\n");
            fprintf(HeaderFile, "#ifndef %s\n", Config->FileFormatC.HeaderGuard);
            fprintf(HeaderFile, "#define %s\n", Config->FileFormatC.HeaderGuard);
            fprintf(HeaderFile, "\n#include <stdint.h>\n");
            fprintf(HeaderFile, "\nextern const ");
            NND__PRINT_STRUCT_DEF(HeaderFile, IType, QScalarInvType, WeightType);
            fprintf(HeaderFile, " %s;", VariableName);
            fprintf(HeaderFile, "\n#endif /* %s */", Config->FileFormatC.HeaderGuard);
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
                Nnd__SerializeValueToCSource(SourceFile, Config->InfoType, NND_TYPE_INT32, &NodeCount[i]);
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
            Nnd__SerializeValueToCSource(SourceFile, Config->InfoType, NND_TYPE_INT32, &InputCount);
            fprintf(SourceFile, ",\n    .LayerCount = ");
            Nnd__SerializeValueToCSource(SourceFile, Config->InfoType, NND_TYPE_INT32, &LayerCount);
            fprintf(SourceFile, ",\n    .QScalarInvShifted = ");
            Nnd__SerializeValueToCSource(SourceFile, Config->InfoType, NND_TYPE_FP32, &QScalarInvShifted);
            fprintf(SourceFile, ",\n    .QScalarInvShamt = ");
            Nnd__SerializeValueToCSource(SourceFile, Config->InfoType, NND_TYPE_INT32, &FxpShift);
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
    const int *NodeCount, int32_t LayerCount,
    const float *Weights, int32_t WeightCount,
    nnd_serialize_config *Config, 
    float QScalarInvShifted, int32_t FxpShift
) {
    const char *ErrorMessage = NULL;
    switch (Config->FileFormat)
    {
    case NND_FILE_FORMAT_C:
    {
        ErrorMessage = Nnd__SerializeFp32ToCSource(
            InputCount,
            NodeCount, LayerCount, 
            Weights, WeightCount,
            Config, QScalarInvShifted, FxpShift
        );
    } break;
    }
    return ErrorMessage;
}

#endif



