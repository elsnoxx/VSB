using System.Security.Cryptography.X509Certificates;
using Antlr4.Runtime;
using Antlr4.Runtime.Misc;

namespace Projekt
{
    public class TypeCheckingVisitor : PLCProjectBaseVisitor<DataType>
    {
        public SymbolTable symbolTable = new SymbolTable();
        public List<string> Errors { get; } = new List<string>();

        // --- Pomocná metoda pro přidávání chyb ---
        private void AddError(ParserRuleContext context, string msg)
        {
            Errors.Add($"Line {context.Start.Line}:{context.Start.Column} - {msg}");
        }

        // --- Program (root) ---
        public override DataType VisitPAREN(PLCProjectParser.PARENContext context)
        {
            return Visit(context.expr());
        }

        // --- Literály ---
        public override DataType VisitINT(PLCProjectParser.INTContext context) => DataType.Int;
        public override DataType VisitFLOAT(PLCProjectParser.FLOATContext context) => DataType.Float;
        public override DataType VisitBOOL(PLCProjectParser.BOOLContext context) => DataType.Bool;
        public override DataType VisitSTR(PLCProjectParser.STRContext context) => DataType.String;

        // --- Proměnná (použití ve výrazu) ---
        public override DataType VisitVARID(PLCProjectParser.VARIDContext context)
        {
            DataType type = symbolTable.GetType(context.GetText());
            if (type == DataType.Error)
                AddError(context, $"Variable '{context.GetText()}' was not declared.");
            return type;
        }

        // --- Aritmetika, Konkatenace (+, -, .) ---
        public override DataType VisitADD_SUB_CONCAT([NotNull] PLCProjectParser.ADD_SUB_CONCATContext context)
        {
            DataType left = Visit(context.expr(0));
            DataType right = Visit(context.expr(1));

            int opType = context.op.Type;

            if (opType == PLCProjectLexer.CONCAT) // Operátor '.'
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
        public override DataType VisitMUL_DIV_MOD([NotNull] PLCProjectParser.MUL_DIV_MODContext context)
        {
            DataType left = Visit(context.expr(0));
            DataType right = Visit(context.expr(1));
            int opType = context.op.Type;

            if (opType == PLCProjectLexer.MOD) // Modulo '%' vyžaduje striktně Int
            {
                if (left == DataType.Int && right == DataType.Int) return DataType.Int;
                AddError(context, "Operator '%' requires two integers.");
                return DataType.Error;
            }

            // Násobení a dělení
            if (left == DataType.Int && right == DataType.Int) return DataType.Int;
            if ((left == DataType.Int || left == DataType.Float) && (right == DataType.Int || right == DataType.Float))
                return DataType.Float;

            AddError(context, $"Operator '{context.op.Text}' requires numeric types.");
            return DataType.Error;
        }

        // --- Deklarace proměnných ---
        public override DataType VisitDeclaration(PLCProjectParser.DeclarationContext context)
        {
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

        // -- Příkazy pro čtení (read) ---
        public override DataType VisitCMDREAD(PLCProjectParser.CMDREADContext context)
        {
            foreach (var id in context.VARID())
            {
                if (symbolTable.GetType(id.GetText()) == DataType.Error)
                    AddError(context, $"Variable '{id.GetText()}' was not declared.");
            }
            return DataType.Error;
        }

        // --- Příkaz pro zápis (write) ---
        public override DataType VisitCMDWRITE(PLCProjectParser.CMDWRITEContext context)
        {
            foreach (var expr in context.expr())
            {
                if (Visit(expr) == DataType.Error)
                    AddError(context, "Invalid expression in 'write' command.");
            }
            return DataType.Error;
        }

        // --- Přiřazení (x = expr) ---
        public override DataType VisitASSIGN(PLCProjectParser.ASSIGNContext context)
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
        public override DataType VisitIF_ELSE(PLCProjectParser.IF_ELSEContext context)
        {
            if (Visit(context.expr()) != DataType.Bool)
                AddError(context, "Condition in 'if' must be a boolean expression.");

            Visit(context.statement(0));
            if (context.statement(1) != null) Visit(context.statement(1));

            return DataType.Error;
        }


        // --- Unární mínus ---
        public override DataType VisitMINUS(PLCProjectParser.MINUSContext context)
        {
            DataType val = Visit(context.expr());

            if (val == DataType.Int || val == DataType.Float)
                return val;

            AddError(context, "Unary '-' requires numeric operand.");
            return DataType.Error;
        }

        #region Logické operátory (AND, OR, NOT, <, >, ==, !=)

        // --- Relace (>, <) ---
        public override DataType VisitREL(PLCProjectParser.RELContext context)
        {
            DataType left = Visit(context.expr(0));
            DataType right = Visit(context.expr(1));

            if ((left == DataType.Int || left == DataType.Float) &&
                (right == DataType.Int || right == DataType.Float))
            {
                return DataType.Bool;
            }

            AddError(context, $"Operator '{context.op.Text}' requires numeric operands.");
            return DataType.Error;
        }

        // --- Rovnost (==, !=) ---
        public override DataType VisitEQUAL(PLCProjectParser.EQUALContext context)
        {
            DataType left = Visit(context.expr(0));
            DataType right = Visit(context.expr(1));

            // stejné typy
            if (left == right)
                return DataType.Bool;

            // int ↔ float je povoleno
            if ((left == DataType.Int && right == DataType.Float) ||
                (left == DataType.Float && right == DataType.Int))
                return DataType.Bool;

            AddError(context, $"Cannot compare {left} and {right}.");
            return DataType.Error;
        }

        // --- Logický NOT ---
        public override DataType VisitNOT(PLCProjectParser.NOTContext context)
        {
            var val = Visit(context.expr());

            if (val == DataType.Bool)
                return DataType.Bool;

            AddError(context, "Operator '!' requires boolean operand.");
            return DataType.Error;
        }

        // --- Logické AND ---
        public override DataType VisitAND(PLCProjectParser.ANDContext context)
        {
            DataType left = Visit(context.expr(0));
            DataType right = Visit(context.expr(1));

            if (left == DataType.Bool && right == DataType.Bool)
                return DataType.Bool;

            AddError(context, "Operator '&&' requires two boolean operands.");
            return DataType.Error;
        }

        // --- Logické OR ---
        public override DataType VisitOR(PLCProjectParser.ORContext context)
        {
            DataType left = Visit(context.expr(0));
            DataType right = Visit(context.expr(1));

            if (left == DataType.Bool && right == DataType.Bool)
                return DataType.Bool;

            AddError(context, "Operator '||' requires two boolean operands.");
            return DataType.Error;
        }

        #endregion


        // --- Cyklus while ---
        public override DataType VisitWHILE(PLCProjectParser.WHILEContext context)
        {
            if (Visit(context.expr()) != DataType.Bool)
                AddError(context, "Condition in 'while' must be a boolean expression.");

            Visit(context.statement());
            return DataType.Error;
        }

        // --- Procedure ---

        public override DataType VisitProcedure(PLCProjectParser.ProcedureContext context)
        {
            string procName = context.VARID().GetText();
            if (!symbolTable.Declare(procName, DataType.Procedure))
                AddError(context, $"Procedure '{procName}' already declared.");

            foreach (var stmt in context.statement()) Visit(stmt);
            return DataType.Procedure;
        }

        public override DataType VisitCALL(PLCProjectParser.CALLContext context)
        {
            if (symbolTable.GetType(context.VARID().GetText()) != DataType.Procedure)
                AddError(context, $"'{context.VARID().GetText()}' is not a procedure.");

            return DataType.Procedure;
        }
    }
}