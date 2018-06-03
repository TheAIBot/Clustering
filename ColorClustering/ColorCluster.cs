using ImageInfo;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Clustering
{
    public class ColorCluster
    {
        public readonly LabPixel ClusterColor;
        public readonly int ClusterSize = 0;
        public readonly PointF CenterPoint;
        public readonly Point TopLeftPoint;
        public readonly Point BottomRightPoint;

        public ColorCluster(LabPixel color, int size, PointF center, Point topLeft, Point bottomRight)
        {
            this.ClusterColor = color;
            this.ClusterSize = size;
            this.CenterPoint = center;
            this.TopLeftPoint = topLeft;
            this.BottomRightPoint = bottomRight;
        }

        public bool IsPure(List<ColorCluster> clusters)
        {
            return clusters.Any(x => x != this && !(x.CenterPoint.X > TopLeftPoint.X &&
                                                    x.CenterPoint.X < BottomRightPoint.X &&
                                                    x.CenterPoint.Y > TopLeftPoint.Y &&
                                                    x.CenterPoint.Y < BottomRightPoint.Y));
        }
    }
}
