namespace TestApp
{
	internal class Places
	{
		private List<Place> places = new List<Place>();


		public void Add(Place place)
		{
			places.Add(place);
		}

		public void Save(string path)
		{
			using (StreamWriter sw = new StreamWriter(path))
			{
				foreach (var place in places)
				{
					sw.WriteLine(place);
				}
			}
		}


		public IEnumerable<T> Filter<T>()
		{
			foreach (var place in places)
			{
				if (place is T x)
				{
					yield return x;
				}
			}
		}

		internal void Sort()
		{
			places.Sort(new PlaceComparer());
		}
	}
}
