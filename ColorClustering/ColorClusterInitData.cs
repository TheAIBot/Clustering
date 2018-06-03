using ImageInfo;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Clustering
{
    public class ColorClusterInitData
    {
        public int R;
        public int G;
        public int B;
        public int X;
        public int Y;
        public int ClusterSize;
        public int MinX = int.MaxValue;
        public int MaxX;
        public int MinY = int.MaxValue;
        public int MaxY;

        public ColorCluster GetColorCluster()
        {
            LabPixel pixel = new RGBPixel(R / ClusterSize, G / ClusterSize, B / ClusterSize).ToLabPixel();

            PointF center = new PointF((float)X / ClusterSize, (float)Y / ClusterSize);
            Point topLeft = new Point(MinX, MinY);
            Point bottomRight = new Point(MaxX, MaxY);

            return new ColorCluster(pixel, ClusterSize, center, topLeft, bottomRight);
        }
    }
}
