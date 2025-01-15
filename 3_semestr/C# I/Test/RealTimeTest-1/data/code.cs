Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;

string path = "data.json";
 
// TODO: kód pro načtení dat
// var jsonData = ....;
 
Places places = new Places();
 
foreach (var element in jsonData.Elements)
{
    var place = element.ToPlace();
    if (place != null)
    {
        places.Add(place);
    }
}
 
places.Sort();

// TODO: volání metody Filter pro získání všech vrcholů.
// var peaks = ....

foreach(var peak in peaks)
{
    Console.WriteLine(peak);
}
places.Save("places.txt");