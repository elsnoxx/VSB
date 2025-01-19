namespace TestApp
{
	internal class Element
	{


		public long id { get; set; }
		public double Lat { get; set; }
		public double Lon { get; set; }
		public Dictionary<string, string> Tags { get; set; }


		public Place ToPlace()
		{
			if (this.Tags.ContainsKey("natural"))
			{
				return new Peak()
				{
					Elevation = this.Tags.ContainsKey("ele") ? double.Parse(this.Tags["ele"]) : null,
					Name = this.Tags.ContainsKey("name") ? this.Tags["name"] : null,
					Lat = this.Lat,
					Lon = this.Lon
				};
			}
			if (this.Tags.ContainsKey("shop"))
			{
				return new Shop()
				{
					Name = this.Tags.ContainsKey("name") ? this.Tags["name"] : null,
					Type = this.Tags["shop"],
					OpeningHours = this.Tags.ContainsKey("opening_hours") ? this.Tags["opening_hours"] : "???",
					Lat = this.Lat,
					Lon = this.Lon
				};
			}
			return null;
		}
	}
}
