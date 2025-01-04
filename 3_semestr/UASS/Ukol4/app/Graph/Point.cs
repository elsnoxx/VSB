using System;

namespace Network
{
    public class Point
    {
        public int X { get; set; }
        public int Y { get; set; }
        public DateTime Timestamp { get; set; }

        public Point(int x, int y, long timestamp)
        {
            this.X = x;
            this.Y = y;
            this.Timestamp = DateTimeOffset.FromUnixTimeSeconds(timestamp).DateTime;
        }
    }
}
