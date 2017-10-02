﻿using ImageInfo;
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
        public readonly Point CenterPoint;

        public ColorCluster(LabPixel color, int size, Point center)
        {
            this.ClusterColor = color;
            this.ClusterSize = size;
            this.CenterPoint = center;
        }
    }
}
