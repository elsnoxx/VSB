using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Ukol1
{
    public struct Brick
    {
        public int x;
        public int y;

        public Brick(int x, int y)
        {
            this.x = x;
            this.y = y;
        }
    }

    public struct CompositeBrick
    {
        public int x;
        public int y;
        List<Brick> bricks;

        public CompositeBrick(int x, int y, List<Brick> bricks)
        {
            this.x = x;
            this.y = y;
            this.bricks = bricks;
        }

        // Metoda pro posunutí složené kostičky dolů
        public void MoveDown()
        {
            this.y++; // Posuneme složenou kostičku dolů
        }

        // Metoda pro získání absolutních souřadnic jednotlivých bricks
        public List<Brick> GetAbsoluteBricks()
        {
            List<Brick> absoluteBricks = new List<Brick>();
            foreach (var brick in bricks)
            {
                absoluteBricks.Add(new Brick(brick.x + x, brick.y + y));
            }
            return absoluteBricks;
        }
    }
    internal class StructureArray
    {
        private List<CompositeBrick> compositeBricks = new List<CompositeBrick>(); // Seznam složených kostiček

        private const int width = 40;  // Šířka herního pole
        private const int height = 20; // Výška herního pole

        // Metoda pro přidání nové kostičky do hry
        public void AddCompositeBrick(CompositeBrick compositeBrick)
        {
            compositeBricks.Add(compositeBrick);
        }

        // Metoda pro vytvoření kostičky tvaru T
        public CompositeBrick CreateTBrick(int x, int y)
        {
            List<Brick> bricks = new List<Brick>
            {
                new Brick(0, 0), // Levý díl T
                new Brick(1, 0), // Střední díl T
                new Brick(2, 0), // Pravý díl T
                new Brick(1, 1)  // Střední dolní díl T
            };
            return new CompositeBrick(x, y, bricks);
        }

        // Metoda pro vytvoření kostičky tvaru Z
        public CompositeBrick CreateZBrick(int x, int y)
        {
            List<Brick> bricks = new List<Brick>
            {
                new Brick(0, 0), // Levý horní díl Z
                new Brick(1, 0), // Pravý horní díl Z
                new Brick(1, 1), // Levý dolní díl Z
                new Brick(2, 1)  // Pravý dolní díl Z
            };
            return new CompositeBrick(x, y, bricks);
        }

        // Metoda pro posun všech kostiček dolů
        public void MoveAllBricksDown()
        {
            foreach (var compositeBrick in compositeBricks)
            {
                compositeBrick.MoveDown();
            }
        }

        // Metoda pro vykreslení scény
        public void PrintScene()
        {
            char[,] scene = new char[height, width];

            // Inicializace prázdné scény
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    scene[i, j] = ' '; // Prázdné místo
                }
            }

            // Přidání kostiček do scény
            foreach (var compositeBrick in compositeBricks)
            {
                List<Brick> absoluteBricks = compositeBrick.GetAbsoluteBricks();
                foreach (var brick in absoluteBricks)
                {
                    if (brick.y >= 0 && brick.y < height && brick.x >= 0 && brick.x < width)
                    {
                        scene[brick.y, brick.x] = '\u2588'; // Plný čtverec
                    }
                }
            }

            // Vykreslení scény do konzole
            Console.SetCursorPosition(0, 0);
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    Console.Write(scene[i, j]);
                    Console.Write(scene[i, j]); // Dvojité vykreslení pro lepší viditelnost čtverců
                }
                Console.WriteLine();
            }
        }

        // Hlavní smyčka hry
        public void Run()
        {
            AddCompositeBrick(CreateTBrick(5, 0)); // Přidání kostičky T
            AddCompositeBrick(CreateZBrick(10, 0)); // Přidání kostičky Z

            while (true)
            {
                PrintScene();  // Vykreslení scény
                MoveAllBricksDown();  // Posun všech kostiček dolů
                Thread.Sleep(1000);  // Pauza na 1 sekundu
            }
        }
    }
}
