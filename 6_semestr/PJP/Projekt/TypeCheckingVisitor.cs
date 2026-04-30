using Antlr4.Runtime;
using Antlr4.Runtime.Misc;

namespace Projekt
{
    public class TypeCheckingVisitor : PLCBaseVisitor<DataType>
    {
        private SymbolTable symbolTable = new SymbolTable();
        public List<string> Errors { get; } = new List<string>();

        private void AddError(ParserRuleContext context, string msg)
        {
            Errors.Add($"Line {context.Start.Line}:{context.Start.Column} - {msg}");
        }

        // --- Literály ---
        public override DataType VisitINT(PLCParser.INTContext context) => DataType.Int;
        public override DataType VisitFLOAT(PLCParser.FLOATContext context) => DataType.Float;
        public override DataType VisitBOOL(PLCParser.BOOLContext context) => DataType.Bool;
        public override DataType VisitSTR(PLCParser.STRContext context) => DataType.String;

        // --- Proměnná (použití ve výrazu) ---
        public override DataType VisitVARID(PLCParser.VARIDContext context)
        {
            DataType type = symbolTable.GetType(context.GetText());
            if (type == DataType.Error)
                AddError(context, $"Variable '{context.GetText()}' was not declared.");
            return type;
        }

        // --- Aritmetika, Konkatenace (+, -, .) ---
        // V nové gramatice se metoda jmenuje podle labelu # ADD_SUB_CONCAT
        public override DataType VisitADD_SUB_CONCAT([NotNull] PLCParser.ADD_SUB_CONCATContext context)
        {
            DataType left = Visit(context.expr(0));
            DataType right = Visit(context.expr(1));

            // V nové gramatice používáme typy tokenů z Lexeru
            int opType = context.op.Type;

            if (opType == PLCLexer.CONCAT) // Operátor '.'
            {
                if (left == DataType.String && right == DataType.String) return DataType.String;
                AddError(context, "Operator '.' can only be used with two strings.");
                return DataType.Error;
            }

            // Klasické sčítání/odčítání
            if (left == DataType.Int && right == DataType.Int) return DataType.Int;
            if ((left == DataType.Int || left == DataType.Float) && (right == DataType.Int || right == DataType.Float))
                return DataType.Float;

            AddError(context, $"Binary operator '{context.op.Text}' cannot be used with {left} and {right}.");
            return DataType.Error;
        }

        // --- Násobení, Dělení, Modulo (*, /, %) ---
        public override DataType VisitMUL_DIV_MOD([NotNull] PLCParser.MUL_DIV_MODContext context)
        {
            DataType left = Visit(context.expr(0));
            DataType right = Visit(context.expr(1));
            int opType = context.op.Type;

            if (opType == PLCLexer.MOD) // Modulo '%' vyžaduje striktně Int
            {
                if (left == DataType.Int && right == DataType.Int) return DataType.Int;
                AddError(context, "Operator '%' requires two integers.");
                return DataType.Error;
            }

            // Násobení a dělení (podobně jako sčítání)
            if (left == DataType.Int && right == DataType.Int) return DataType.Int;
            if ((left == DataType.Int || left == DataType.Float) && (right == DataType.Int || right == DataType.Float))
                return DataType.Float;

            AddError(context, $"Operator '{context.op.Text}' requires numeric types.");
            return DataType.Error;
        }

        // --- Deklarace proměnných ---
        public override DataType VisitCMDVAR(PLCParser.CMDVARContext context)
        {
            // context.vartype() vrací pravidlo, které obsahuje INT_KW, FLOAT_KW atd.
            string typeStr = context.vartype().GetText().ToLower();
            DataType type = typeStr switch
            {
                "int" => DataType.Int,
                "float" => DataType.Float,
                "bool" => DataType.Bool,
                "string" => DataType.String,
                _ => DataType.Error
            };

            foreach (var id in context.VARID())
            {
                if (!symbolTable.Declare(id.GetText(), type))
                    AddError(context, $"Variable '{id.GetText()}' is already declared.");
            }
            return DataType.Error;
        }

        // --- Přiřazení (x = expr) ---
        public override DataType VisitASSIGN(PLCParser.ASSIGNContext context)
        {
            DataType varType = symbolTable.GetType(context.VARID().GetText());
            DataType exprType = Visit(context.expr());

            if (varType == DataType.Error) return DataType.Error;

            // Kontrola: Float do Int nepůjde
            if (varType == DataType.Int && exprType == DataType.Float)
            {
                AddError(context, $"Cannot assign Float value to Int variable '{context.VARID().GetText()}'.");
                return DataType.Error;
            }

            // Povoleno: Stejné typy NEBO Int do Float (automatický cast)
            if (varType == exprType || (varType == DataType.Float && exprType == DataType.Int))
                return varType;

            AddError(context, $"Type mismatch: cannot assign {exprType} to {varType}.");
            return DataType.Error;
        }

        // --- Podmínky a cykly ---
        public override DataType VisitIF_ELSE(PLCParser.IF_ELSEContext context)
        {
            if (Visit(context.expr()) != DataType.Bool)
                AddError(context, "Condition in 'if' must be a boolean expression.");

            Visit(context.statement(0));
            if (context.statement(1) != null) Visit(context.statement(1));

            return DataType.Error;
        }

        public override DataType VisitWHILE(PLCParser.WHILEContext context)
        {
            if (Visit(context.expr()) != DataType.Bool)
                AddError(context, "Condition in 'while' must be a boolean expression.");

            Visit(context.statement());
            return DataType.Error;
        }
    }
}