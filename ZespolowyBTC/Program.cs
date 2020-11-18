using System;
using System.Drawing;
using System.Drawing.Imaging;

namespace ZespolowyBTC
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            //TestReadWriteBMP();

            BTCAlgorithm.Prototype(4);
        }

        static void TestReadWriteBMP()
        {
            Bitmap bitmap = (Bitmap)Image.FromFile(@"D:\Repos\zespolowy-btc\TestImages\camera256.bmp");

            bitmap.Save(@"D:\Repos\zespolowy-btc\TestImages\_czyDziala.bmp", ImageFormat.Bmp);

            //int width = bitmap.Width;
            //int height = bitmap.Height;
            Color pixel00 = bitmap.GetPixel(1, 10);
            Console.WriteLine(pixel00.ToString());

            bitmap.Dispose();


            // Drawing and saving BMP
            Bitmap bmp = new Bitmap(400, 500);
            Graphics gBmp = Graphics.FromImage(bmp);
            gBmp.DrawEllipse(new Pen(Color.Red), 40f, 40f, 60f, 80f);
            gBmp.Dispose();
            bmp.Save(@"D:\Repos\zespolowy-btc\TestImages\_testowyObrazek.bmp", ImageFormat.Bmp);

            // Bitmap.LockBits()
        }
    }
}
