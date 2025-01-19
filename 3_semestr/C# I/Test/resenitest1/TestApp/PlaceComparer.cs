namespace TestApp
{
	internal class PlaceComparer : IComparer<Place>
	{
		public int Compare(Place x, Place y)
		{
			if (x is Peak p1 && y is Peak p2)
			{
				double ele1 = (p1.Elevation ?? 0);
				double ele2 = (p2.Elevation ?? 0);
				if (ele1 < ele2)
				{
					return 1;
				}
				if (ele1 > ele2)
				{
					return -1;
				}
				return x.Name.CompareTo(y.Name);
			}
			if (x is Peak)
			{
				return -1;
			}
			if (y is Peak)
			{
				return 1;
			}
			return x.Name.CompareTo(y.Name);
		}
	}
}
