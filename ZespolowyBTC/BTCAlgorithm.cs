using Emgu.CV;
using Numpy;
using System;
using System.IO;
using System.Reflection;

namespace ZespolowyBTC
{
    class BTCAlgorithm
    {
        public static void Prototype(int blockSize)
        {
            Console.WriteLine(Path.GetDirectoryName(Assembly.GetEntryAssembly().Location));
            var image = CvInvoke.Imread(@"../../../../TestImages/lennagrey.bmp");

            Mat imageGrey = new Mat();
            //NDarray imageGrey = new NDarray(typeof(double), image.Width, image.Height);
            CvInvoke.CvtColor(image, imageGrey, Emgu.CV.CvEnum.ColorConversion.Bgr2Gray);
            var imageBtc = np.zeros(imageGrey.Width, imageGrey.Height);
            //var block = np.zeros(blockSize, blockSize);

            //var ndImage = np.zeros(imageGrey.Width, imageGrey.Height);
            //imageGrey.CopyTo<NDarray>(ndImage);
            //imageGrey.GetData
            NDarray data = imageGrey.GetData();

            for(int w = 0; w < imageGrey.Width; w += blockSize)
            {
                for(int h = 0; h < imageGrey.Height; h += blockSize)
                {

                    //for(int i = 0; i < 4; i++)
                    //{
                    //    for(int j = 0; j < 4; j++)
                    //    {
                    //        block[i, j] = imageGrey.Da[w + i, h + j];
                    //    }
                    //}
                    //byte[] data = new byte[imageGrey.Width * imageGrey.Height];
                    //imageGrey.CopyTo<byte>(data);
                    //;
                    var block = data[$"{w}:{w + blockSize}, {h}:{h + blockSize}"];
                    //block = imageGrey.data[$"{w}: {w + blockSize}, {h}: {h + blockSize}"];
                    //Console.WriteLine(block.ToString());
                    var mean = np.mean(block);
                    var std = np.std(block);
                    var q = np.sum(block > mean);
                    var a = mean - (std * MathF.Sqrt(q / (blockSize ^ 2 - q)));
                    var b = q == 0 ? mean + (std * MathF.Sqrt((blockSize ^ 2 - q) / q)) : a;
                    block = a.where
                    Console.WriteLine(mean.ToString());
                    Console.WriteLine("dupa");

                    //NDarray ndImageGrey = new NDarray(imageGrey.GetDataPointer(), imageGrey., np.uint8);//new NDarray(imageGrey.GetData());
                    //imageGrey.GetArray(out byte[] plainArray); //there it is, c# array for nparray constructor
                    //NDarray nDarray = np.array(plainArray, dtype: np.uint8); //party party
                    //block = imageGrey;

                    //var ndImageGrey = np.asarray(imageGrey.DataPointer);
                    //block = ndImageGrey[$"{w}: {w + blockSize}, {h}: {h + blockSize}"];
                    //Console.WriteLine(ndImageGrey);
                    //var mean = np.mean(block);

                    //Console.WriteLine($"DUPA FOR H {mean}");
                }
            }

            //var imageBtc = np.zeros()
        }
    }
}
