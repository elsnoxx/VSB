# Základní praktiky v C#

## 1. Práce s třídami a vlastnostmi

```csharp
public class Book
{
    public int Id { get; set; }
    public string Title { get; set; }
    public string Author { get; set; }
    public string ISBN { get; set; }
}
```
- **Použití:** Vytvoření instance:  
  `var book = new Book { Title = "1984", Author = "Orwell" };`

---

## 2. Volání privátních metod a práce s reflexí

```csharp
// Získání privátní metody a její zavolání
var storage = new DataStorage();
var method = typeof(DataStorage).GetMethod("Insert", BindingFlags.NonPublic | BindingFlags.Instance);
int id = (int)method.Invoke(storage, new object[] { book });
```
- **Použití:** Získání a zavolání privátní metody.

---

### Další příklady práce s reflexí

#### Získání a nastavení hodnoty privátního pole

```csharp
var field = typeof(Book).GetField("_internalCode", BindingFlags.NonPublic | BindingFlags.Instance);
field.SetValue(book, "ABC123");
string value = (string)field.GetValue(book);
```
- **Použití:** Čtení a zápis do privátního pole.

---

#### Získání všech vlastností třídy

```csharp
var props = typeof(Book).GetProperties();
foreach (var prop in props)
{
    Console.WriteLine($"{prop.Name} = {prop.GetValue(book)}");
}
```
- **Použití:** Výpis všech vlastností a jejich hodnot.

---

#### Dynamické vytvoření instance třídy

```csharp
var type = Type.GetType("Namespace.Book");
var instance = Activator.CreateInstance(type);
```
- **Použití:** Vytvoření instance třídy podle názvu.

---

#### Získání všech metod třídy

```csharp
var methods = typeof(Book).GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
foreach (var m in methods)
{
    Console.WriteLine(m.Name);
}
```
- **Použití:** Výpis všech metod třídy.

---

#### Dynamické nastavení hodnoty vlastnosti podle jména

```csharp
var prop = typeof(Book).GetProperty("Title");
prop.SetValue(book, "Nový název");
```
- **Použití:** Nastavení hodnoty vlastnosti podle jejího jména.

---

#### Kontrola, zda třída obsahuje určitou vlastnost/metodu

```csharp
bool hasProp = typeof(Book).GetProperty("Title") != null;
bool hasMethod = typeof(Book).GetMethod("ToString") != null;
```
- **Použití:** Ověření existence vlastnosti nebo metody.

---

## 3. Asynchronní programování

```csharp
public async Task<string> DownloadAsync(string url)
{
    using var client = new HttpClient();
    return await client.GetStringAsync(url);
}
```
- **Použití:**  
  `string xml = await DownloadAsync(url);`

---

## 4. Práce s XML a LINQ to XML

```csharp
var doc = XDocument.Parse(xmlString);
var items = doc.Descendants("POLOZKA")
    .Select(x => new
    {
        Chodnota = x.Element("CHODNOTA")?.Value,
        Text = x.Element("TEXT")?.Value
    }).ToList();
```
- **Použití:** Zpracování XML z webové služby.

---

## 5. Validace vstupů

```csharp
if (string.IsNullOrWhiteSpace(email) || !email.Contains("@"))
{
    throw new ArgumentException("Neplatný email");
}
```
- **Použití:** Kontrola vstupních dat ve formuláři.

---

## 6. Serializace objektu do textového souboru (pouze string vlastnosti)

```csharp
var sb = new StringBuilder();
foreach (var prop in obj.GetType().GetProperties())
{
    if (prop.PropertyType == typeof(string))
    {
        sb.AppendLine($"#{prop.Name} => {prop.GetValue(obj)};");
    }
}
File.WriteAllText("data.txt", sb.ToString());
```
- **Použití:** Uložení dat do souboru pomocí reflexe.

---

## 7. Práce s SQLite (desktopová aplikace)

```csharp
using var conn = new SqliteConnection("Data Source=company.db");
conn.Open();
var cmd = conn.CreateCommand();
cmd.CommandText = "INSERT INTO Company (Nazev, Dic, Obec, Poznamka) VALUES ($nazev, $dic, $obec, $poznamka)";
cmd.Parameters.AddWithValue("$nazev", nazev);
cmd.Parameters.AddWithValue("$dic", dic);
cmd.Parameters.AddWithValue("$obec", obec);
cmd.Parameters.AddWithValue("$poznamka", poznamka);
cmd.ExecuteNonQuery();
```
- **Použití:** Uložení dat do SQLite databáze.

---

## 8. Volání webového API a zpracování JSON

```csharp
using var client = new HttpClient();
string json = await client.GetStringAsync(apiUrl);
dynamic data = JsonConvert.DeserializeObject(json);
string name = data.obchodniJmeno;
```
- **Použití:** Získání dat z API a práce s JSON.

---

## 9. Základní validace formulářových polí v MVC

```csharp
[Required]
[StringLength(500)]
public string Poznamka { get; set; }
```
- **Použití:** Validace v modelech pro ASP.NET MVC.

---

## 10. Základní asynchronní metoda pro načtení dat

```csharp
public async Task<List<Book>> GetBooksAsync()
{
    // ... načtení dat
}
```
- **Použití:**  
  `var books = await GetBooksAsync();`

---

## 11. Práce s HttpClient a XML API

```csharp
using var client = new HttpClient();
string xml = await client.GetStringAsync(apiUrl);
// Dále zpracovat pomocí XDocument
```

---

## 12. Základní práce s kolekcemi

```csharp
var books = new List<Book>();
books.Add(new Book { Title = "Test" });
var found = books.FirstOrDefault(b => b.Id == 1);
```

---

## 13. Základní použití async/await v událostech (např. kliknutí na tlačítko)

```csharp
private async void Button_Click(object sender, EventArgs e)
{
    await SomeAsyncMethod();
}
```

---

## 14. Základní práce s MVC kontrolerem

```csharp
public IActionResult Index()
{
    var books = storage.GetAll();
    return View(books);
}
```

---

## 15. Přidání vlastního servisu do ASP.NET Core MVC

```csharp
// 1. Definujte rozhraní a implementaci servisu
public interface IBookService
{
    void AddBook(Book book);
    IEnumerable<Book> GetAll();
}

public class BookService : IBookService
{
    private readonly List<Book> _books = new();
    public void AddBook(Book book) => _books.Add(book);
    public IEnumerable<Book> GetAll() => _books;
}

// 2. Registrace servisu v Program.cs nebo Startup.cs
builder.Services.AddSingleton<IBookService, BookService>();

// 3. Použití servisu v kontroleru (dependency injection)
public class BooksController : Controller
{
    private readonly IBookService _bookService;
    public BooksController(IBookService bookService)
    {
        _bookService = bookService;
    }

    public IActionResult Index()
    {
        var books = _bookService.GetAll();
        return View(books);
    }
}
```
- **Použití:** Vlastní servis přidáte do DI kontejneru a následně jej využijete v kontroleru.

---

## 16. Přidání vlastního middleware do ASP.NET Core

```csharp
// 1. Vytvoření middleware třídy
public class LoggingMiddleware
{
    private readonly RequestDelegate _next;
    public LoggingMiddleware(RequestDelegate next)
    {
        _next = next;
    }

    public async Task InvokeAsync(HttpContext context)
    {
        Console.WriteLine($"Request: {context.Request.Method} {context.Request.Path}");
        await _next(context);
    }
}

// 2. Registrace middleware v Program.cs nebo Startup.cs
app.UseMiddleware<LoggingMiddleware>();
```
- **Použití:** Middleware zaregistrujete v pipeline a můžete tak například logovat nebo upravovat požadavky/odpovědi.

---

## Postup: Práce s externím a přeloženým DDL (assembly) v C# pomocí reflexe

Pokud dostanete externí .dll knihovnu (například bez zdrojového kódu), můžete s ní pracovat pomocí reflexe. Zde je krok za krokem postup:

1. **Přidejte DDL do projektu**
   - Zkopírujte .dll soubor do složky projektu (např. `libs\MyLibrary.dll`).
   - V Solution Explorer klikněte pravým tlačítkem na „References“ → „Add Reference...“ → „Browse“ a vyberte vaši .dll.

2. **Načtěte assembly v kódu**
   ```csharp
   var assembly = Assembly.LoadFrom("libs/MyLibrary.dll");
   ```

3. **Získejte seznam typů v assembly**
   ```csharp
   var types = assembly.GetTypes();
   foreach (var type in types)
   {
       Console.WriteLine(type.FullName);
   }
   ```

4. **Najděte konkrétní třídu**
   ```csharp
   var myType = assembly.GetType("Namespace.ClassName");
   ```

5. **Vytvořte instanci třídy**
   ```csharp
   var instance = Activator.CreateInstance(myType);
   ```

6. **Získejte a zavolejte veřejnou nebo privátní metodu**
   ```csharp
   var method = myType.GetMethod("MethodName", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
   var result = method.Invoke(instance, new object[] { /* parametry */ });
   ```

7. **Získejte a nastavte hodnotu pole nebo vlastnosti**
   ```csharp
   var prop = myType.GetProperty("PropertyName");
   prop.SetValue(instance, "hodnota");
   var value = prop.GetValue(instance);
   ```

8. **Získání privátního pole**
   ```csharp
   var field = myType.GetField("_privateField", BindingFlags.NonPublic | BindingFlags.Instance);
   field.SetValue(instance, 123);
   var val = field.GetValue(instance);
   ```

---

### Shrnutí
- Přidejte .dll do projektu a načtěte ji pomocí `Assembly.LoadFrom`.
- Najděte požadovaný typ (třídu) podle jména.
- Vytvořte instanci a pracujte s metodami, vlastnostmi a poli pomocí reflexe.
- Vše lze provádět i bez znalosti zdrojového kódu, pouze na základě názvů členů.

---

## Postup: Vytvoření HTTP requestu a zpracování odpovědi jako JSON v C#

1. **Přidejte NuGet balíček**
   - Ujistěte se, že máte v projektu balíček `Newtonsoft.Json` (nebo použijte `System.Text.Json` v .NET Core 3+).

2. **Vytvořte HttpClient**
   ```csharp
   using var client = new HttpClient();
   ```

3. **Odešlete HTTP GET požadavek**
   ```csharp
   string url = "https://api.example.com/data";
   string json = await client.GetStringAsync(url);
   ```

4. **Definujte model pro deserializaci**
   ```csharp
   public class MyData
   {
       public string Name { get; set; }
       public int Value { get; set; }
   }
   ```

5. **Deserializujte JSON do objektu**
   - Pomocí Newtonsoft.Json:
     ```csharp
     var data = JsonConvert.DeserializeObject<MyData>(json);
     ```
   - Nebo pomocí System.Text.Json:
     ```csharp
     var data = JsonSerializer.Deserialize<MyData>(json);
     ```

6. **Práce s načtenými daty**
   ```csharp
   Console.WriteLine(data.Name);
   Console.WriteLine(data.Value);
   ```

---

### Shrnutí kroků
- Přidejte potřebný NuGet balíček.
- Vytvořte HttpClient a odešlete požadavek.
- Získejte odpověď jako string (JSON).
- Definujte model podle struktury JSON.
- Deserializujte JSON do objektu.
- Pracujte s daty v objektu.

---

**Tip:** Pro POST/PUT požadavky použijte metody `PostAsync`/`PutAsync` a serializujte data do JSON pomocí `JsonConvert.SerializeObject` nebo `JsonSerializer.Serialize`.

## Postup: Vytvoření HTTP requestu a zpracování odpovědi jako XML v C#

1. **Vytvořte HttpClient**
   ```csharp
   using var client = new HttpClient();
   ```

2. **Odešlete HTTP GET požadavek**
   ```csharp
   string url = "https://api.example.com/data.xml";
   string xml = await client.GetStringAsync(url);
   ```

3. **Načtěte XML do XDocument**
   ```csharp
   var doc = XDocument.Parse(xml);
   ```

4. **Vyhledávání v XML pomocí LINQ to XML**
   - **Výběr všech elementů určitého typu:**
     ```csharp
     var items = doc.Descendants("Item");
     foreach (var item in items)
     {
         string name = item.Element("Name")?.Value;
         string value = item.Element("Value")?.Value;
         Console.WriteLine($"{name}: {value}");
     }
     ```
   - **Výběr podle atributu:**
     ```csharp
     var selected = doc.Descendants("Item")
                       .FirstOrDefault(x => (string)x.Attribute("id") == "123");
     if (selected != null)
     {
         Console.WriteLine(selected);
     }
     ```
   - **Výběr hodnoty z vnořeného elementu:**
     ```csharp
     string specialValue = doc.Descendants("Item")
                              .Where(x => (string)x.Element("Name") == "Special")
                              .Select(x => (string)x.Element("Value"))
                              .FirstOrDefault();
     ```

5. **Deserializace XML do objektu (volitelné)**
   ```csharp
   [XmlRoot("Item")]
   public class Item
   {
       public string Name { get; set; }
       public string Value { get; set; }
   }

   // Deserializace jednoho objektu
   var serializer = new XmlSerializer(typeof(Item));
   using var reader = new StringReader(xml);
   var item = (Item)serializer.Deserialize(reader);
   ```

---

### Shrnutí kroků
- Vytvořte HttpClient a odešlete požadavek.
- Získejte odpověď jako string (XML).
- Načtěte XML do XDocument.
- Vyhledávejte v XML pomocí LINQ to XML (elementy, atributy, hodnoty).
- (Volitelně) Deserializujte XML do objektu pomocí XmlSerializer.

---

**Tip:** Pro složitější XML struktury je vhodné použít LINQ to XML, pro jednoduché mapování objektů použijte `XmlSerializer`.