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

        public ColorCluster GetColorCluster()
        {
            LabPixel pixel = new RGBPixel(R / ClusterSize, G / ClusterSize, B / ClusterSize).ToLabPixel();

            PointF center = new PointF((float)X / ClusterSize, (float)Y / ClusterSize);

            return new ColorCluster(pixel, ClusterSize, center);
        }
    }
}
