using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Ukol1
{
    internal class MultidimensionalArray
    {
        bool[,] MultiDim = new bool[40, 40];

        public void Run()
        {
            Random rnd = new Random();
            for (int i = 0; i < MultiDim.GetLength(0); i++)
            {
                for (int j = 0; j < MultiDim.GetLength(1); j++)
                {
                    MultiDim[i, j] = false;
                }
            }

            for (int i = 0; i < 4; i++)
            {
                int x = rnd.Next(40);
                int y = rnd.Next(40);
                CreateZBlock(x, y, MultiDim);
            }

            for (int i = 0; i < 4; i++)
            {
                int x = rnd.Next(40);
                int y = rnd.Next(40);
                CreateTBlock(x, y, MultiDim);
            }


            MainLoop(MultiDim);
        }
        public void MainLoop(bool[,] MultiDim)
        {
            int time_sleep = 1000;
            Console.ForegroundColor = ConsoleColor.Black;
            Console.BackgroundColor = ConsoleColor.White;
            Console.Clear();
            while (true)
            {
                Console.SetCursorPosition(0, 0);
                Print();
                for (int i = MultiDim.GetLength(0) - 1; i >= 0; i--)
                {
                    for (int j = 0; j < MultiDim.GetLength(1); j++)
                    {
                        if (MultiDim[i, j])
                        {
                            if (i == MultiDim.GetLength(0) - 1)
                            {
                                MultiDim[i, j] = false;
                            }
                            else
                            {
                                // Posun hodnoty dolů
                                MultiDim[i + 1, j] = MultiDim[i, j];
                                MultiDim[i, j] = false; // Vymazání původního místa
                            }
                        }
                    }
                }
                for (int j = 0; j < MultiDim.GetLength(1); j++)
                {
                    Console.Write('-');
                    Console.Write('-');
                }
                Thread.Sleep(time_sleep - 5);
            }
        }

        public void CreateTBlock(int x, int y, bool[,] MultiDim)
        {
            // Zkontrolujeme, zda kostička nepřesáhne hranice pole
            if (x >= 0 && x + 1 < MultiDim.GetLength(0) && y >= 0 && y + 2 < MultiDim.GetLength(1))
            {
                // Tvar T (vodorovná část)
                MultiDim[x, y] = true;
                MultiDim[x, y + 1] = true;
                MultiDim[x, y + 2] = true;

                // Tvar T (svislá část)
                MultiDim[x + 1, y + 1] = true;
                MultiDim[x + 2, y + 1] = true;
            }
        }

        public void CreateZBlock(int x, int y, bool[,] MultiDim)
        {
            // Zkontrolujeme, zda kostička nepřesáhne hranice pole
            if (x >= 0 && x + 1 < MultiDim.GetLength(0) && y >= 0 && y + 2 < MultiDim.GetLength(1))
            {
                // Tvar Z (horní část)
                MultiDim[x, y] = true;
                MultiDim[x, y + 1] = true;

                // Tvar Z (spodní část)
                MultiDim[x + 1, y + 1] = true;
                MultiDim[x + 1, y + 2] = true;
            }
        }
        public void Print()
        {

            for (int i = 0; i < MultiDim.GetLength(0); i++)
            {
                for (int j = 0; j < MultiDim.GetLength(1); j++)
                {
                    if (MultiDim[i, j])
                    {
                        Console.Write('\u2588');
                        Console.Write('\u2588');
                    }
                    else
                    {
                        Console.Write(' ');
                        Console.Write(' ');
                    }

                }
                Console.WriteLine();
            }
        }
    }
}
