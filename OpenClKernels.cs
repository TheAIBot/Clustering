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
kernel void RGBToLab(constant uchar* rgbPixels, global char* labPixels, const float maxColorNumber)
{
    const int index = get_global_id(0) * 3;

    const float red   = convert_float(rgbPixels[index + 0]);
    const float green = convert_float(rgbPixels[index + 1]);
    const float blue  = convert_float(rgbPixels[index + 2]);


    //First convert from RGB to XYZ
    const float sR = red   / maxColorNumber;
    const float sG = green / maxColorNumber;
    const float sB = blue  / maxColorNumber;

    const float rLinear = (sR > 0.04045) ? pow((sR + 0.055) / 1.055, 2.4) : sR / 12.92;
    const float gLinear = (sG > 0.04045) ? pow((sG + 0.055) / 1.055, 2.4) : sG / 12.92;
    const float bLinear = (sB > 0.04045) ? pow((sB + 0.055) / 1.055, 2.4) : sB / 12.92;

    const float X = rLinear * 0.4124 + gLinear * 0.3576 + bLinear * 0.1805;
    const float Y = rLinear * 0.2126 + gLinear * 0.7152 + bLinear * 0.0722;
    const float Z = rLinear * 0.0193 + gLinear * 0.1192 + bLinear * 0.9505;


    //Then convert from XYZ to Lab
    const float xRef = 0.95047;
    const float yRef = 1.00;
    const float zRef = 1.08883;

    const float xReffed = X / xRef;
    const float yReffed = Y / yRef;
    const float zReffed = Z / zRef;

    const float xF = (xReffed > 0.008856) ? pow(xReffed, 1 / 3.0f) : (7.787 * xReffed) + (4 / 29.0);
    const float yF = (yReffed > 0.008856) ? pow(yReffed, 1 / 3.0f) : (7.787 * yReffed) + (4 / 29.0);
    const float zF = (zReffed > 0.008856) ? pow(zReffed, 1 / 3.0f) : (7.787 * zReffed) + (4 / 29.0);

    const float L = 116 * yF - 16;
    const float a = 500 * (xF - yF);
    const float b = 200 * (yF - zF);


    //Now convert to char
    const int iL = convert_int(L);
    const int ia = convert_int(a);
    const int ib = convert_int(b);

    const int rangedL = clamp(iL, -128, 127);
    const int rangeda = clamp(ia, -128, 127);
    const int rangedb = clamp(ib, -128, 127);

    const char cL = convert_char(rangedL);
    const char ca = convert_char(rangeda);
    const char cb = convert_char(rangedb);

    labPixels[index + 0] = cL;
    labPixels[index + 1] = ca;
    labPixels[index + 2] = cb;
}";

        private const string LabDistances = @"
float DistanceCIE94(const float L1, const float a1, const float b1, const float L2, const float a2, const float b2)
{
    const float C1 = sqrt((a1 * a1) + (b1 * b1));
    const float C2 = sqrt((a2 * a2) + (b2 * b2));
    const float DeltaCab = C1 - C2;

    const float DeltaL = L1 - L2;
    const float Deltaa = a1 - a2;
    const float Deltab = b1 - b2;

    const float beforeDeltaHab = (Deltaa * Deltaa) + (Deltab * Deltab) - (DeltaCab * DeltaCab);
    const float DeltaHab = (beforeDeltaHab < 0) ? 0 : sqrt(beforeDeltaHab);

    const float kL = 1;
    const float kC = 1;
    const float kH = 1;
    const float K1 = 0.045;
    const float K2 = 0.015;

    const float SL = 1;
    const float SC = 1 + K1 * C1;
    const float SH = 1 + K2 * C1;

    const float LRes = DeltaL / (kL * SL);
    const float CRes = DeltaCab / (kC * SC);
    const float HRes = DeltaHab / (kH * SH);

    const float beforeReturn = (LRes * LRes) + (CRes * CRes) + (HRes * HRes);
    return beforeReturn < 0 ? 0 : sqrt(beforeReturn);
}

uchar IsDistanceLessThan(const float cL, const float ca, const float cb, constant char* labPixels, const int index, const float allowedDistance)
{
    const float L = convert_float(labPixels[index + 0]);
    const float a = convert_float(labPixels[index + 1]);
    const float b = convert_float(labPixels[index + 2]);
    const float distance = DistanceCIE94(cL, ca, cb, L, a, b);
    return distance <= allowedDistance;
}

kernel void LabDistances(constant char* labPixels, global uchar* labDistances, const int bigWidth, const int bigHeight, const float allowedDistance)
{
    const int index = get_global_id(0);

    const int x = (index % bigWidth);
    const int y = (index / bigWidth);

    const int labDistancesIndex = (y * bigWidth + x) * 4;

    const int centerIndex = (y * bigWidth + x) * 3;
    const float centerL = convert_float(labPixels[centerIndex + 0]);
    const float centera = convert_float(labPixels[centerIndex + 1]);
    const float centerb = convert_float(labPixels[centerIndex + 2]);

    //MY GOD THESE INDEXES ARE UGLY
    //But it works!
    const int topIndex = (max((y - 1), 0) * bigWidth + x) * 3;
    const int leftIndex = (y * bigWidth + max((x - 1), 0)) * 3;
    const int rightIndex = (y * bigWidth + min((x + 1), bigWidth - 1)) * 3;
    const int bottomIndex = (min((y + 1), bigHeight - 1) * bigWidth + x) * 3;

    labDistances[labDistancesIndex + 0] = IsDistanceLessThan(centerL, centera, centerb, labPixels, topIndex   , allowedDistance);
    labDistances[labDistancesIndex + 1] = IsDistanceLessThan(centerL, centera, centerb, labPixels, leftIndex  , allowedDistance);
    labDistances[labDistancesIndex + 2] = IsDistanceLessThan(centerL, centera, centerb, labPixels, rightIndex , allowedDistance);
    labDistances[labDistancesIndex + 3] = IsDistanceLessThan(centerL, centera, centerb, labPixels, bottomIndex, allowedDistance);
}";

        private const string RemoveNoise = @"
int LabDistanceSum(global uchar* labDistances, const int x, const int y, const int width, const int height)
{
    const int index = (clamp(y, 0, height - 1) * width + clamp(x, 0, width - 1)) * 4;
    return convert_int(labDistances[index + 0]) + 
           convert_int(labDistances[index + 1]) + 
           convert_int(labDistances[index + 2]) + 
           convert_int(labDistances[index + 3]);
}

kernel void RemoveNoise(global uchar* labDistances, const int width, const int height)
{
    const int index = get_global_id(0);
    const int centerX = index % width;
    const int centerY = index / width;

    const int RANGE = 4;
    const float MAX_SUM = convert_float((RANGE * 2 + 1) * (RANGE * 2 + 1) * 4);
    const float MIN_PERCENT = 0.9f;
    const int MIN_NOISE = convert_int(MAX_SUM * MIN_PERCENT);

    int sum = 0;
    for(int yOffset = -RANGE; yOffset <= RANGE; yOffset++)
    {
        for(int xOffset = -RANGE; xOffset <= RANGE; xOffset++)
        {
            sum += LabDistanceSum(labDistances, centerX + xOffset, centerY + yOffset, width, height);
        }
    }
    
    labDistances[index * 4 + 0] = (sum > MIN_NOISE) ? 1 : labDistances[index * 4 + 0];
    labDistances[index * 4 + 1] = (sum > MIN_NOISE) ? 1 : labDistances[index * 4 + 1];
    labDistances[index * 4 + 2] = (sum > MIN_NOISE) ? 1 : labDistances[index * 4 + 2];
    labDistances[index * 4 + 3] = (sum > MIN_NOISE) ? 1 : labDistances[index * 4 + 3];
}";

        public const string EdgeDetector = @"
kernel void DetectEdges(global uchar* labDistances, const int width, const int height)
{
    const int index = get_global_id(0);
}";

        public const string GaussianBlur = @"
constant int GAUSS_RADIUS = 3;
constant int GAUSS_WIDTH = 7;
constant float GAUSS_VALUES[] = 
{
    0.00000067f, 0.00002292f, 0.00019117f, 0.00038771f, 0.00019117f, 0.00002292f, 0.00000067f, 
    0.00002292f, 0.00078634f, 0.00655965f, 0.01330373f, 0.00655965f, 0.00078633f, 0.00002292f, 
    0.00019117f, 0.00655965f, 0.05472157f, 0.11098164f, 0.05472157f, 0.00655965f, 0.00019117f, 
    0.00038771f, 0.01330373f, 0.11098164f, 0.22508352f, 0.11098164f, 0.01330373f, 0.00038771f, 
    0.00019117f, 0.00655965f, 0.05472157f, 0.11098164f, 0.05472157f, 0.00655965f, 0.00019117f, 
    0.00002292f, 0.00078633f, 0.00655965f, 0.01330373f, 0.00655965f, 0.00078633f, 0.00002292f, 
    0.00000067f, 0.00002292f, 0.00019117f, 0.00038771f, 0.00019117f, 0.00002292f, 0.00000067f
};

kernel void GaussianBlur(constant uchar* pixels, global uchar* gaussedPixels, const int width, const int height)
{
    const int index = get_global_id(0);
    const int x = (index % width);
    const int y = (index / width);

    float redSum   = 0;
    float greenSum = 0;
    float blueSum  = 0;
    for(int yOffset = 0; yOffset < GAUSS_WIDTH; yOffset++)
    {
        for(int xOffset = 0; xOffset < GAUSS_WIDTH; xOffset++)
        {
            const int rangedX = clamp(x + xOffset - GAUSS_RADIUS, 0, width  - 1);
            const int rangedY = clamp(y + yOffset - GAUSS_RADIUS, 0, height - 1);
            const int pixelIndex = (rangedY * width + rangedX) * 3;
    
            const float weight = GAUSS_VALUES[yOffset * GAUSS_WIDTH + xOffset];
    
            redSum   += convert_float(pixels[pixelIndex + 0]) * weight;
            greenSum += convert_float(pixels[pixelIndex + 1]) * weight;
            blueSum  += convert_float(pixels[pixelIndex + 2]) * weight;
        }
    }

    gaussedPixels[index * 3 + 0] = convert_uchar(redSum);
    gaussedPixels[index * 3 + 1] = convert_uchar(greenSum);
    gaussedPixels[index * 3 + 2] = convert_uchar(blueSum);
}";

        public const string ClusterDetector = @"

int BestCluster(const int x, const int y, const int width, constant char* labDistances)
{
    const int topIndex    = (x + 0) + (y - 1) * width;
    const int bottomIndex = (x + 0) + (y + 1) * width;
    const int leftIndex   = (x - 1) + (y + 0) * width;
    const int rightIndex  = (x + 1) + (y + 0) * width;

    const int topDistance    = convert_int(labDistances[topIndex]);
    const int bottomDistance = convert_int(labDistances[bottomIndex]);
    const int leftDistance   = convert_int(labDistances[leftIndex]);
    const int rightDistance  = convert_int(labDistances[rightIndex]);

    const int topCluster    = clusterMap[topIndex];
    const int bottomCluster = clusterMap[bottomIndex];
    const int leftCluster   = clusterMap[leftIndex];
    const int rightCluster  = clusterMap[rightIndex];

    const int topResult    = topDistance    * topCluster;
    const int bottomResult = bottomDistance * bottomCluster;
    const int leftResult   = leftDistance   * leftCluster;
    const int rightResult  = rightDistance  * rightCluster;

    return min(topResult, min(bottomResult, min(leftResult, rightResult)))
}

kernel void ClusterDetector(constant char* labDistances, global int* clusterMap, const int width, const int height, const int boxWidth, const int boxHeight)
{
    const int index = get_global_id(0);
    const int correctedIndex = index * (boxWidth * boxHeight);
    const int correctWidth = width - 2;
    const int x = (correctedIndex % correctWidth) + 1;
    const int y = (correctedIndex / correctWidth) + 1;

    for(int dx = 0; dx < boxWidth; dx++)
    {
        for(int dy = 0; dy < boxHeight; dy++)
        {
            clusterMap[(x + dx) + (y + dy) * correctWidth] = BestCluster(x + dx ,y + dy, correctWidth, labDistances);
        }
    }

    for(int dx = boxWidth - 1; dx >= 0; dx--)
    {
        for(int dy = boxHeight - 1; dy >= 0; dy--)
        {
            clusterMap[(x + dx) + (y + dy) * correctWidth] = BestCluster(x + dx ,y + dy, correctWidth, labDistances);
        }
    }
}";

        public const string Kernel = RGBToLab + LabDistances + RemoveNoise/* + EdgeDetector*/ + GaussianBlur;// + ClusterDetector;
    }
}
