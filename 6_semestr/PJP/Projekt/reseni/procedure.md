Task procedury

## 1. Gramatika (PLCProject.g4)Do gramatiky přidáme definici procedury pomocí klíčového slova void a příkaz pro její volání.Fragment kódu// Přidat do pravidla statement

```
statement 
    : command? SEMI                                   # CMD
    | LBRACE statement* RBRACE                        # BLOCK
    | IF LPAREN expr RPAREN statement (ELSE statement)? # IF_ELSE
    | WHILE LPAREN expr RPAREN statement              # WHILE
    | VOID VARID LPAREN RPAREN LBRACE statement* RBRACE # PROC_DEF // Definice
    | VARID LPAREN RPAREN SEMI                        # PROC_CALL // Volání
    ;

// Přidat do Lexeru
VOID : 'void';

```
## 2. Symbol Table a Datové typy (DataType.cs, SymbolTable.cs)Musíme evidovat, že určité jméno nepatří proměnné, ale proceduře.  

DataType.cs: Přidej Procedure.  
SymbolTable.cs: Metody zůstávají stejné, jen budeme ukládat DataType.Procedure.  

## 3. Type Checking (TypeCheckingVisitor.cs)Zde musíme pohlídat, aby nedošlo ke kolizi jmen a aby se volaly jen existující procedury.  

```C#
public override DataType VisitPROC_DEF(PLCProjectParser.PROC_DEFContext context)
{
    string name = context.VARID().GetText();
    if (!symbolTable.Declare(name, DataType.Procedure))
        AddError(context, $"Name '{name}' is already used.");
    
    foreach (var stmt in context.statement()) Visit(stmt);
    return DataType.Error;
}

public override DataType VisitPROC_CALL(PLCProjectParser.PROC_CALLContext context)
{
    string name = context.VARID().GetText();
    if (symbolTable.GetType(name) != DataType.Procedure)
        AddError(context, $"Procedure '{name}' is not defined.");
    return DataType.Error;
}
```

## 4. Code Generation (CodeGeneratorVisitor.cs)Kód procedury musí být uvozen návěstím a ukončen instrukcí pro návrat (ret). Volání provedeme instrukcí call.  
```C#
public override string VisitPROC_DEF(PLCProjectParser.PROC_DEFContext context)
{
    string name = context.VARID().GetText();
    StringBuilder sb = new StringBuilder();
    
    // Procedura musí být "mimo" hlavní tok, nebo ji musíme přeskočit
    string skipLabel = $"skip_{name}";
    sb.AppendLine($"jmp {skipLabel}"); 
    sb.AppendLine($"label {name}");
    
    foreach (var stmt in context.statement())
        sb.AppendLine(Visit(stmt));
    
    sb.AppendLine("ret"); // Instrukce pro návrat
    sb.AppendLine($"label {skipLabel}");
    
    return sb.ToString();
}

public override string VisitPROC_CALL(PLCProjectParser.PROC_CALLContext context)
{
    return $"call {context.VARID().GetText()}";
}
```

## 5. Virtuální stroj (VirtualMachine.cs)VM potřebuje tzv. Call Stack, aby věděla, kam se po skončení procedury vrátit. 
``` C#
public class VirtualMachine
{
    private Stack<object> stack = new Stack<object>();
    private Stack<int> callStack = new Stack<int>(); // Zásobník pro návratové adresy
    // ... ostatní fieldy stejné ...

    public void Run()
    {
        int ip = 0;
        while (ip < instructions.Count)
        {
            var inst = instructions[ip];
            string cmd = inst[0].ToLower();

            switch (cmd)
            {
                case "call":
                    callStack.Push(ip + 1); // Ulož adresu příští instrukce
                    ip = labels[inst[1]];   // Skoč na návěstí procedury
                    continue;

                case "ret":
                    if (callStack.Count > 0)
                    {
                        ip = callStack.Pop(); // Vrať se zpět
                        continue;
                    }
                    break;
                
                // ... ostatní case instrukce (push, pop, add, atd.) ...
            }
            ip++;
        }
    }
}
```

Jak to bude fungovat v praxi:Když napíšeš v PLC:Fragment kódu
```
void pozdrav() {
  write "Ahoj";
}
pozdrav();
```