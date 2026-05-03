1. Gramatika (PLCProject.g4)Musíme přidat deklaraci pole s velikostí a přístup k prvkům přes index.  Fragment kódu
```
// Úprava deklarace proměnných (přidání volitelné velikosti pole)
command 
    : vartype VARID (LBRACKET INT RBRACKET)? (COMMA VARID (LBRACKET INT RBRACKET)?)* # CMDVAR
    | ... ;

// Úprava výrazů (přístup k indexu) a přiřazení do pole
expr
    : ...
    | VARID LBRACKET expr RBRACKET ASSIGN expr        # ARRAY_ASSIGN
    | VARID LBRACKET expr RBRACKET                   # ARRAY_ACCESS
    | ... ;

LBRACKET : '[' ;
RBRACKET : ']' ;
```

## 2. Symbol Table a Datové typy (DataType.cs, SymbolTable.cs)Musíme rozlišit, zda je identifikátor běžná proměnná nebo pole konkrétního typu.  DataType.cs: Můžeš přidat typy jako IntArray, FloatArray atd., nebo mít v SymbolTable příznak, že jde o pole.  SymbolTable.cs: Uprav tak, aby si ukládala i informaci o délce pole (pro kontrolu mezí).  3. Type Checking (TypeCheckingVisitor.cs)Zde musíme hlídat, aby index byl vždy typu Int.

```C#
public override DataType VisitARRAY_ACCESS(PLCProjectParser.ARRAY_ACCESSContext context) {
    DataType varType = symbolTable.GetType(context.VARID().GetText());
    if (Visit(context.expr()) != DataType.Int) 
        AddError(context, "Array index must be an integer.");
    return varType; // Vrací základní typ pole (např. Int, pokud je to IntArray)
}
```
## 4. Code Generation (CodeGeneratorVisitor.cs)Podle tvého obrázku musíme generovat specifické instrukce.  

```C#
public override string VisitCMDVAR(PLCProjectParser.CMDVARContext context) {
    StringBuilder sb = new StringBuilder();
    // Pro každý VARID v deklaraci zkontrolujeme, zda má [INT]
    // Pokud ano, vygenerujeme:
    // push I {velikost}
    // createarray {název}
    return sb.ToString();
}

public override string VisitARRAY_ASSIGN(PLCProjectParser.ARRAY_ASSIGNContext context) {
    StringBuilder sb = new StringBuilder();
    sb.AppendLine(Visit(context.expr(1))); // Hodnota k uložení
    sb.AppendLine(Visit(context.expr(0))); // Index
    sb.AppendLine($"arraysave {context.VARID().GetText()}");
    return sb.ToString();
}
```

## 5. Virtuální stroj (VirtualMachine.cs)Ve VM musíme změnit způsob ukládání do paměti. Pole bude reprezentováno jako C# pole (object[]) uložené v memory pod daným názvem.  C#// Přidat do switch(cmd) v metodě Run():

```C#
case "createarray":
    int size = (int)stack.Pop();
    memory[inst[1]] = new object[size]; // Alokace pole v paměti VM
    break;

case "arraysave":
    int saveIdx = (int)stack.Pop();
    object value = stack.Pop();
    ((object[])memory[inst[1]])[saveIdx] = value; // Uložení na index[cite: 12]
    break;

case "arrayload":
    int loadIdx = (int)stack.Pop();
    stack.Push(((object[])memory[inst[1]])[loadIdx]); // Načtení z indexu na zásobník[cite: 12]
    break;
```