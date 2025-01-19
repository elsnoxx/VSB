namespace TestApp
{
	internal class Shop : Place
	{
		public string OpeningHours { get; set; }
		public string Type { get; set; }

		public override string ToString()
		{
			return $"OBCHOD | {Name} ({Lat}, {Lon}): {Type} - {OpeningHours}";
		}
	}
}
