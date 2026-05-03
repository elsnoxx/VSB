using System.Text;
using Antlr4.Runtime.Misc;

namespace Projekt
{
    public class CodeGeneratorVisitor : PLCProjectBaseVisitor<string>
    {
        private TypeCheckingVisitor typeChecker;
        private int labelCounter = 0;
        private readonly SymbolTable symbolTable;

        public CodeGeneratorVisitor(TypeCheckingVisitor typeCheckingVisitor)
        {
            this.typeChecker = typeCheckingVisitor;
            this.symbolTable = typeCheckingVisitor.symbolTable;
        }
        private string GetNextLabel() => $"L{labelCounter++}";

        // ---------------- PROGRAM ----------------
        public override string VisitProg(PLCProjectParser.ProgContext context)
        {
            StringBuilder sb = new StringBuilder();

            foreach (var stmt in context.statement())
            {

                sb.AppendLine(Visit(stmt));
            }

            return sb.ToString();
        }

        // ---------------- LITERALS ----------------
        public override string VisitINT(PLCProjectParser.INTContext context)
        => $"push I {context.GetText()}";

        public override string VisitFLOAT(PLCProjectParser.FLOATContext context)
        => $"push F {context.GetText()}";

        public override string VisitBOOL(PLCProjectParser.BOOLContext context)
        => $"push B {context.GetText().ToLower()}";

        public override string VisitSTR(PLCProjectParser.STRContext context)
        => $"push S {context.GetText()}";

        public override string VisitVARID(PLCProjectParser.VARIDContext context)
            => $"load {context.GetText()}";

        // ---------------- ARITHMETIC ----------------
        public override string VisitMUL_DIV_MOD(PLCProjectParser.MUL_DIV_MODContext context)
        {
            DataType leftType = typeChecker.Visit(context.expr(0));
            DataType rightType = typeChecker.Visit(context.expr(1));
            DataType resultType = typeChecker.Visit(context);

            StringBuilder sb = new StringBuilder();

            // Levý operand + případný cast na float
            sb.AppendLine(Visit(context.expr(0)));
            if (resultType == DataType.Float && leftType == DataType.Int) sb.AppendLine("itof");

            // Pravý operand + případný cast na float
            sb.AppendLine(Visit(context.expr(1)));
            if (resultType == DataType.Float && rightType == DataType.Int) sb.AppendLine("itof");

            string suffix = (resultType == DataType.Float) ? "F" : "I";
            string op = context.op.Type switch
            {
                PLCProjectLexer.MUL => $"mul {suffix}",
                PLCProjectLexer.DIV => $"div {suffix}",
                PLCProjectLexer.MOD => "mod",
                _ => ""
            };
            sb.AppendLine(op);
            return sb.ToString().TrimEnd();
        }

        public override string VisitADD_SUB_CONCAT(PLCProjectParser.ADD_SUB_CONCATContext context)
        {
            DataType leftType = typeChecker.Visit(context.expr(0));
            DataType rightType = typeChecker.Visit(context.expr(1));
            DataType resultType = typeChecker.Visit(context); // Celkový typ operace

            StringBuilder sb = new StringBuilder();

            // Levá strana + případný cast
            sb.AppendLine(Visit(context.expr(0)));
            if (resultType == DataType.Float && leftType == DataType.Int) sb.AppendLine("itof");

            // Pravá strana + případný cast
            sb.AppendLine(Visit(context.expr(1)));
            if (resultType == DataType.Float && rightType == DataType.Int) sb.AppendLine("itof");

            // Samotná operace
            if (context.op.Type == PLCProjectLexer.CONCAT) sb.AppendLine("concat");
            else
            {
                string suffix = (resultType == DataType.Float) ? "F" : "I";
                string op = (context.op.Type == PLCProjectLexer.ADD) ? "add" : "sub";
                sb.AppendLine($"{op} {suffix}");
            }

            return sb.ToString();
        }

        // ---------------- ASSIGN ----------------
        public override string VisitASSIGN(PLCProjectParser.ASSIGNContext context)
        {
            string varName = context.VARID().GetText();
            return $"{Visit(context.expr())}\nsave {varName}\nload {varName}";
        }

        // ---------------- IF / ELSE ----------------
        public override string VisitIF_ELSE(PLCProjectParser.IF_ELSEContext context)
        {
            string elseLabel = GetNextLabel();
            string endLabel = GetNextLabel();

            StringBuilder sb = new StringBuilder();

            sb.AppendLine(Visit(context.expr()));
            sb.AppendLine($"fjmp {elseLabel}");

            sb.AppendLine(Visit(context.statement(0)));
            sb.AppendLine($"jmp {endLabel}");

            sb.AppendLine($"label {elseLabel}");

            if (context.statement().Length > 1)
                sb.AppendLine(Visit(context.statement(1)));

            sb.AppendLine($"label {endLabel}");

            return sb.ToString();
        }

        // ---------------- WHILE ----------------
        public override string VisitWHILE(PLCProjectParser.WHILEContext context)
        {
            string startLabel = GetNextLabel();
            string endLabel = GetNextLabel();

            StringBuilder sb = new StringBuilder();

            sb.AppendLine($"label {startLabel}");
            sb.AppendLine(Visit(context.expr()));
            sb.AppendLine($"fjmp {endLabel}");
            sb.AppendLine(Visit(context.statement()));
            sb.AppendLine($"jmp {startLabel}");
            sb.AppendLine($"label {endLabel}");

            return sb.ToString();
        }

        // ---------------- WRITE ----------------
        public override string VisitCMDWRITE(PLCProjectParser.CMDWRITEContext context)
        {
            StringBuilder sb = new StringBuilder();

            foreach (var expr in context.expr())
            {
                sb.AppendLine(Visit(expr));
            }

            sb.AppendLine($"print {context.expr().Length}");
            return sb.ToString();
        }

        // ---------------- READ ----------------
        public override string VisitCMDREAD(PLCProjectParser.CMDREADContext context)
        {
            StringBuilder sb = new StringBuilder();
            foreach (var id in context.VARID())
            {
                string varName = id.GetText();
                // Získej typ přímo z tabulky symbolů, kterou sdílíš s TypeCheckerem
                DataType type = symbolTable.GetType(varName);

                string suffix = type switch
                {
                    DataType.Int => "I",
                    DataType.Float => "F",
                    DataType.String => "S",
                    DataType.Bool => "B",
                    _ => throw new Exception($"Neznámý typ pro proměnnou {varName}")
                };

                sb.AppendLine($"read {suffix}");
                sb.AppendLine($"save {varName}");
            }
            return sb.ToString();
        }

        // ---------------- PAREN ----------------
        public override string VisitPAREN(PLCProjectParser.PARENContext context)
            => Visit(context.expr());

        // ---------------- DEFAULT ----------------
        protected override string AggregateResult(string aggregate, string nextResult)
        {
            if (string.IsNullOrEmpty(aggregate)) return nextResult;
            if (string.IsNullOrEmpty(nextResult)) return aggregate;
            return aggregate + "\n" + nextResult;
        }

        public override string VisitREL(PLCProjectParser.RELContext context)
        {
            // Zjistíme typ levého operandu (zadání říká, že oba mají stejný typ T)
            DataType type = typeChecker.Visit(context.expr(0));
            string suffix = (type == DataType.Float) ? "F" : "I";

            string op = context.op.Type switch
            {
                PLCProjectLexer.LT => $"lt {suffix}",
                PLCProjectLexer.GT => $"gt {suffix}",
                _ => ""
            };
            return $"{Visit(context.expr(0))}\n{Visit(context.expr(1))}\n{op}";
        }

        public override string VisitEQUAL(PLCProjectParser.EQUALContext context)
        {
            // Získání typů operandů pomocí sdíleného typeCheckeru
            DataType leftType = typeChecker.Visit(context.expr(0));
            DataType rightType = typeChecker.Visit(context.expr(1));

            // Určení sufixu pro instrukci eq (podle zadání I, F nebo S)
            string suffix = "I";
            if (leftType == DataType.String || rightType == DataType.String) suffix = "S";
            else if (leftType == DataType.Float || rightType == DataType.Float) suffix = "F";
            else if (leftType == DataType.Bool) suffix = "B";

            StringBuilder sb = new StringBuilder();

            // Generování kódu pro operandy s případným itof
            sb.AppendLine(Visit(context.expr(0)));
            if (suffix == "F" && leftType == DataType.Int) sb.AppendLine("itof");

            sb.AppendLine(Visit(context.expr(1)));
            if (suffix == "F" && rightType == DataType.Int) sb.AppendLine("itof");

            // Pro != (NEQ) vygenerujeme rovnost a pak ji znegujeme pomocí 'not'
            if (context.op.Type == PLCProjectLexer.NEQ)
            {
                sb.AppendLine($"eq {suffix}");
                sb.AppendLine("not");
            }
            else // Pro == (EQ)
            {
                sb.AppendLine($"eq {suffix}");
            }

            return sb.ToString().TrimEnd();
        }

        public override string VisitAND(PLCProjectParser.ANDContext context)
        {
            return $"{Visit(context.expr(0))}\n{Visit(context.expr(1))}\nand";
        }

        public override string VisitOR(PLCProjectParser.ORContext context)
        {
            return $"{Visit(context.expr(0))}\n{Visit(context.expr(1))}\nor";
        }
        public override string VisitNOT(PLCProjectParser.NOTContext context)
        {
            return $"{Visit(context.expr())}\nnot";
        }

        public override string VisitMINUS(PLCProjectParser.MINUSContext context)
        {
            DataType type = typeChecker.Visit(context);
            string suffix = (type == DataType.Float) ? "F" : "I";
            return $"{Visit(context.expr())}\numinus {suffix}";
        }
    }
}