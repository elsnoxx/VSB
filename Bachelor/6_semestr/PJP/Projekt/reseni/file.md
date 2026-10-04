## 1. Gramatika (PLCProject.g4)Musíme přidat klíčová slova, nový datový typ a pravidla pro příkazy fopen a fappend.  Fragment kódu

```
// Přidat do vartype
vartype : ... | FILE_KW ;

// Přidat do command
command
    : ...
    | FOPEN VARID COMMA expr                  # CMDFOPEN
    | FAPPEND VARID (COMMA expr)*             # CMDFAPPEND
    ;

// LEXER
FILE_KW  : 'file' ;
FOPEN    : 'fopen' ;
FAPPEND  : 'fappend' ;
```
## 2. Datové typy a SymbolTable (DataType_3.cs)Do výčtu DataType přidej položku File. V SymbolTable se pak proměnná deklaruje jako file a;, což v tabulce uloží a s typem DataType.File.  

## 3. Generování kódu (CodeGeneratorVisitor_3.cs)Generování musí odpovídat tvému návrhu instrukcí. U fappend je klíčové spočítat počet argumentů (včetně samotného souboru).  

```C#
public override string VisitCMDFOPEN(PLCProjectParser.CMDFOPENContext context)
{
    StringBuilder sb = new StringBuilder();
    sb.AppendLine(Visit(context.expr())); // push S "test.txt"
    sb.AppendLine("fopen");               // instrukce vytvoří stream
    sb.AppendLine($"save {context.VARID().GetText()}"); // uloží stream do 'a'
    return sb.ToString();
}

public override string VisitCMDFAPPEND(PLCProjectParser.CMDFAPPENDContext context)
{
    StringBuilder sb = new StringBuilder();
    string varName = context.VARID().GetText();

    // 1. Načtení streamu (souboru) na zásobník
    sb.AppendLine($"load {varName}");

    // 2. Načtení všech výrazů (dat k zápisu)
    foreach (var expr in context.expr())
    {
        sb.AppendLine(Visit(expr));
    }

    // 3. Instrukce fappend s počtem parametrů (soubor + výrazy)
    int paramCount = context.expr().Length + 1; 
    sb.AppendLine($"fappend {paramCount}");
    
    return sb.ToString();
}
```
## 4. Virtuální stroj (VirtualMachine_3.cs)VM musí umět pracovat s objektem StreamWriter (nebo podobným), který bude uložen v paměti jako typ object.  C#// Do Run() switch(cmd):

```C#
case "fopen":
    string path = stack.Pop().ToString();
    // Vytvoříme StreamWriter a pushneme ho jako objekt
    var sw = new System.IO.StreamWriter(path, append: true);
    sw.AutoFlush = true; 
    stack.Push(sw); 
    break;

case "fappend":
    int count = int.Parse(inst[1]);
    var args = new List<object>();
    for (int i = 0; i < count - 1; i++) args.Add(stack.Pop());
    
    // Poslední na zásobníku (první v pořadí fappend) je stream
    System.IO.StreamWriter fileStream = (System.IO.StreamWriter)stack.Pop();
    
    args.Reverse();
    // Zápis do souboru (podobně jako print)
    fileStream.WriteLine(string.Join(" ", args));
    break;
```