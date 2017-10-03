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

    int topIndex = (max((y - 1), 0) * bigWidth + x) * 3;
    float topL = convert_float(labPixels[topIndex + 0]);
    float topa = convert_float(labPixels[topIndex + 1]);
    float topb = convert_float(labPixels[topIndex + 2]);
    float topDistance = DistanceCIE94(centerL, centera, centerb, topL, topa, topb);
    labDistances[labDistancesIndex + 0] = topDistance <= allowedDistance;

    int leftIndex = (y * bigWidth + max((x - 1), 0)) * 3;
    float leftL = convert_float(labPixels[leftIndex + 0]);
    float lefta = convert_float(labPixels[leftIndex + 1]);
    float leftb = convert_float(labPixels[leftIndex + 2]);
    float leftDistance = DistanceCIE94(centerL, centera, centerb, leftL, lefta, leftb);
    labDistances[labDistancesIndex + 1] = leftDistance <= allowedDistance;

    int rightIndex = (y * bigWidth + min((x + 1), bigWidth - 1)) * 3;
    float rightL = convert_float(labPixels[rightIndex + 0]);
    float righta = convert_float(labPixels[rightIndex + 1]);
    float rightb = convert_float(labPixels[rightIndex + 2]);
    float rightDistance = DistanceCIE94(centerL, centera, centerb, rightL, righta, rightb);
    labDistances[labDistancesIndex + 2] = rightDistance <= allowedDistance;

    int bottomIndex = (min((y + 1), bigHeight - 1) * bigWidth + x) * 3;
    float bottomL = convert_float(labPixels[bottomIndex + 0]);
    float bottoma = convert_float(labPixels[bottomIndex + 1]);
    float bottomb = convert_float(labPixels[bottomIndex + 2]);
    float bottomDistance = DistanceCIE94(centerL, centera, centerb, bottomL, bottoma, bottomb);
    labDistances[labDistancesIndex + 3] = bottomDistance <= allowedDistance;
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
    
    int minCount = 10000;
    for(int yOffset = -4; yOffset <= 4; yOffset++)
    {
        for(int xOffset = -4; xOffset <= 4; xOffset++)
        {
            int newPotentialMin = LabDistanceSum(labDistances, centerX + xOffset, centerY + yOffset, width, height);
            minCount = min(minCount, newPotentialMin);
        }
    }

    labDistances[index * 4 + 0] = (minCount <= 1) ? labDistances[index * 4 + 0] : 1;
    labDistances[index * 4 + 1] = (minCount <= 1) ? labDistances[index * 4 + 1] : 1;
    labDistances[index * 4 + 2] = (minCount <= 1) ? labDistances[index * 4 + 2] : 1;
    labDistances[index * 4 + 3] = (minCount <= 1) ? labDistances[index * 4 + 3] : 1;
}";

        public const string Kernel = RGBToLab + LabDistances + RemoveNoise;
    }
}
