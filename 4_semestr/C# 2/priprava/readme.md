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

## 2. Volání privátních metod pomocí reflexe

```csharp
var storage = new DataStorage();
var method = typeof(DataStorage).GetMethod("Insert", BindingFlags.NonPublic | BindingFlags.Instance);
int id = (int)method.Invoke(storage, new object[] { book });
```
- **Použití:** Získání a zavolání privátní metody.

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

Tento přehled pokrývá základní praktiky, které budete potřebovat pro vaše zadání v C#. Pokud potřebujete konkrétní ukázku pro nějakou část zadání, napište!