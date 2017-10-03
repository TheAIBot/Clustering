using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Clustering
{
    public static class OpenClKernels
    {
        private const string RGBToLab = @"
kernel void RGBToLab(constant uchar* rgbPixels, global char* labPixels, float maxColorNumber)
{
    int index = get_global_id(0) * 3;

    float red   = convert_float(rgbPixels[index + 0]);
    float green = convert_float(rgbPixels[index + 1]);
    float blue  = convert_float(rgbPixels[index + 2]);


    //First convert from RGB to XYZ
    float sR = red   / maxColorNumber;
    float sG = green / maxColorNumber;
    float sB = blue  / maxColorNumber;

    float rLinear = (sR > 0.04045) ? pow((sR + 0.055) / 1.055, 2.4) : sR / 12.92;
    float gLinear = (sG > 0.04045) ? pow((sG + 0.055) / 1.055, 2.4) : sG / 12.92;
    float bLinear = (sB > 0.04045) ? pow((sB + 0.055) / 1.055, 2.4) : sB / 12.92;

    float X = rLinear * 0.4124 + gLinear * 0.3576 + bLinear * 0.1805;
    float Y = rLinear * 0.2126 + gLinear * 0.7152 + bLinear * 0.0722;
    float Z = rLinear * 0.0193 + gLinear * 0.1192 + bLinear * 0.9505;


    //Then convert from XYZ to Lab
    const float xRef = 0.95047;
    const float yRef = 1.00;
    const float zRef = 1.08883;

    float xReffed = X / xRef;
    float yReffed = Y / yRef;
    float zReffed = Z / zRef;

    float xF = (xReffed > 0.008856) ? pow(xReffed, 1 / 3.0f) : (7.787 * xReffed) + (4 / 29.0);
    float yF = (yReffed > 0.008856) ? pow(yReffed, 1 / 3.0f) : (7.787 * yReffed) + (4 / 29.0);
    float zF = (zReffed > 0.008856) ? pow(zReffed, 1 / 3.0f) : (7.787 * zReffed) + (4 / 29.0);

    float L = 116 * yF - 16;
    float a = 500 * (xF - yF);
    float b = 200 * (yF - zF);


    //Now convert to char
    int iL = convert_int(L);
    int ia = convert_int(a);
    int ib = convert_int(b);

    int rangedL = min(max(iL, -128), 127);
    int rangeda = min(max(ia, -128), 127);
    int rangedb = min(max(ib, -128), 127);

    char cL = convert_char(rangedL);
    char ca = convert_char(rangeda);
    char cb = convert_char(rangedb);

    labPixels[index + 0] = cL;
    labPixels[index + 1] = ca;
    labPixels[index + 2] = cb;
}";

        private const string LabDistances = @"
float DistanceCIE94(float L1, float a1, float b1, float L2, float a2, float b2)
{
    float C1 = sqrt((a1 * a1) + (b1 * b1));
    float C2 = sqrt((a2 * a2) + (b2 * b2));
    float DeltaCab = C1 - C2;

    float DeltaL = L1 - L2;
    float Deltaa = a1 - a2;
    float Deltab = b1 - b2;

    float beforeDeltaHab = (Deltaa * Deltaa) + (Deltab * Deltab) - (DeltaCab * DeltaCab);
    float DeltaHab = (beforeDeltaHab < 0) ? 0 : sqrt(beforeDeltaHab);

    const float kL = 1;
    const float kC = 1;
    const float kH = 1;
    const float K1 = 0.045;
    const float K2 = 0.015;

    float SL = 1;
    float SC = 1 + K1 * C1;
    float SH = 1 + K2 * C1;

    float LRes = DeltaL / (kL * SL);
    float CRes = DeltaCab / (kC * SC);
    float HRes = DeltaHab / (kH * SH);

    float beforeReturn = (LRes * LRes) + (CRes * CRes) + (HRes * HRes);
    return beforeReturn < 0 ? 0 : sqrt(beforeReturn);
}

uchar IsDistanceLessThan(float cL, float ca, float cb, constant char* labPixels, int index, float allowedDistance)
{
    float L = convert_float(labPixels[index + 0]);
    float a = convert_float(labPixels[index + 1]);
    float b = convert_float(labPixels[index + 2]);
    float distance = DistanceCIE94(cL, ca, cb, L, a, b);
    return distance <= allowedDistance;
}

kernel void LabDistances(constant char* labPixels, global uchar* labDistances, int bigWidth, int bigHeight, float allowedDistance)
{
    int index = get_global_id(0);

    int x = (index % bigWidth);
    int y = (index / bigWidth);

    int labDistancesIndex = (y * bigWidth + x) * 4;

    int centerIndex = (y * bigWidth + x) * 3;
    float centerL = convert_float(labPixels[centerIndex + 0]);
    float centera = convert_float(labPixels[centerIndex + 1]);
    float centerb = convert_float(labPixels[centerIndex + 2]);

    //MY GOD THESE INDEXES ARE UGLY
    //But it works!
    int topIndex = (max((y - 1), 0) * bigWidth + x) * 3;
    int leftIndex = (y * bigWidth + max((x - 1), 0)) * 3;
    int rightIndex = (y * bigWidth + min((x + 1), bigWidth - 1)) * 3;
    int bottomIndex = (min((y + 1), bigHeight - 1) * bigWidth + x) * 3;

    labDistances[labDistancesIndex + 0] = IsDistanceLessThan(centerL, centera, centerb, labPixels, topIndex   , allowedDistance);
    labDistances[labDistancesIndex + 1] = IsDistanceLessThan(centerL, centera, centerb, labPixels, leftIndex  , allowedDistance);
    labDistances[labDistancesIndex + 2] = IsDistanceLessThan(centerL, centera, centerb, labPixels, rightIndex , allowedDistance);
    labDistances[labDistancesIndex + 3] = IsDistanceLessThan(centerL, centera, centerb, labPixels, bottomIndex, allowedDistance);
}";

        private const string RemoveNoise = @"
int LabDistanceSum(global uchar* labDistances, int x, int y, int width, int height)
{
    int index = (clamp(y, 0, height - 1) * width + clamp(x, 0, width - 1)) * 4;
    return convert_int(labDistances[index + 0]) + 
           convert_int(labDistances[index + 1]) + 
           convert_int(labDistances[index + 2]) + 
           convert_int(labDistances[index + 3]);
}

kernel void RemoveNoise(global uchar* labDistances, int width, int height)
{
    int index = get_global_id(0);
    int centerX = index % width;
    int centerY = index / width;
    const int RANGE = 4;
    const float MAX_SUM = convert_float((RANGE * 2 + 1) * (RANGE * 2 + 1) * 4);
    const float MIN_PERCENT = 0.8f;
    const int MIN_NOISE = convert_int(MAX_SUM * MIN_PERCENT);

    int sum = 0;
    for(int yOffset = -RANGE; yOffset <= RANGE; yOffset++)
    {
        for(int xOffset = -RANGE; xOffset <= RANGE; xOffset++)
        {
            sum += LabDistanceSum(labDistances, centerX + xOffset, centerY + yOffset, width, height);
        }
    }
    
    float percentEqual = convert_float(sum) / MAX_SUM;
    labDistances[index * 4 + 0] = (sum > MIN_NOISE) ? 1 : labDistances[index * 4 + 0];
    labDistances[index * 4 + 1] = (sum > MIN_NOISE) ? 1 : labDistances[index * 4 + 1];
    labDistances[index * 4 + 2] = (sum > MIN_NOISE) ? 1 : labDistances[index * 4 + 2];
    labDistances[index * 4 + 3] = (sum > MIN_NOISE) ? 1 : labDistances[index * 4 + 3];
}";

        public const string Kernel = RGBToLab + LabDistances + RemoveNoise;
    }
}
