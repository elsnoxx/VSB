using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Ukol1
{
    internal class JaggedArray
    {
        bool[][] jaggedArray = new bool[40][];

        public void Run()
        {
            Random rnd = new Random();
            for (int i = 0; i < jaggedArray.GetLength(0); i++)
            {
                jaggedArray[i] = new bool[40];
                for (int j = 0; j < jaggedArray.Length ; j++)
                {
                    jaggedArray[i][j] = false;
                }
            }

            for (int i = 0; i < 4; i++)
            {
                int x = rnd.Next(40);
                int y = rnd.Next(40);
                CreateZBlock(x, y, jaggedArray);
            }

            for (int i = 0; i < 4; i++)
            {
                int x = rnd.Next(40);
                int y = rnd.Next(40);
                CreateTBlock(x, y, jaggedArray);
            }


            MainLoop(jaggedArray);
        }
        public void MainLoop(bool[][] jaggedArray)
        {
            int time_sleep = 1000;
            Console.ForegroundColor = ConsoleColor.Black;
            Console.BackgroundColor = ConsoleColor.White;
            Console.Clear();
            while (true)
            {
                Console.SetCursorPosition(0, 0);
                Print();
                for (int i = jaggedArray.GetLength(0) - 1; i >= 0; i--)
                {
                    for (int j = 0; j < jaggedArray.Length; j++)
                    {
                        if (jaggedArray[i][j])
                        {
                            if (i == jaggedArray.GetLength(0) - 1)
                            {
                                jaggedArray[i][j] = false;
                            }
                            else
                            {
                                // Posun hodnoty dolů
                                jaggedArray[i + 1][ j] = jaggedArray[i][j];
                                jaggedArray[i][j] = false; // Vymazání původního místa
                            }
                        }
                    }
                }
                for (int j = 0; j < jaggedArray.Length; j++)
                {
                    Console.Write('-');
                    Console.Write('-');
                }
                Thread.Sleep(time_sleep - 5);
            }
        }

        public void CreateTBlock(int x, int y, bool[][] jaggedArray)
        {
            // Zkontrolujeme, zda kostička nepřesáhne hranice pole
            if (x >= 0 && x + 1 < jaggedArray.GetLength(0) && y >= 0 && y + 2 < jaggedArray.Length)
            {
                // Tvar T (vodorovná část)
                jaggedArray[x][y] = true;
                jaggedArray[x][y + 1] = true;
                jaggedArray[x][y + 2] = true;

                // Tvar T (svislá část)
                jaggedArray[x + 1][y + 1] = true;
                jaggedArray[x + 2][y + 1] = true;
            }
        }

        public void CreateZBlock(int x, int y, bool[][] jaggedArray)
        {
            // Zkontrolujeme, zda kostička nepřesáhne hranice pole
            if (x >= 0 && x + 1 < jaggedArray.GetLength(0) && y >= 0 && y + 2 < jaggedArray.Length)
            {
                // Tvar Z (horní část)
                jaggedArray[x][y] = true;
                jaggedArray[x][y + 1] = true;

                // Tvar Z (spodní část)
                jaggedArray[x + 1][y + 1] = true;
                jaggedArray[x + 1][y + 2] = true;
            }
        }
        public void Print()
        {

            for (int i = 0; i < jaggedArray.GetLength(0); i++)
            {
                for (int j = 0; j < jaggedArray.Length; j++)
                {
                    if (jaggedArray[i][j])
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
