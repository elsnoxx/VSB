namespace TestApp
{
	internal class Peak : Place
	{
		public double? Elevation { get; set; }

		public override string ToString()
		{
			string ele = Elevation?.ToString() ?? "???";

			return $"VRCHOL | {Name} ({Lat}, {Lon}): {ele}m.n.m";
		}
	}
}
