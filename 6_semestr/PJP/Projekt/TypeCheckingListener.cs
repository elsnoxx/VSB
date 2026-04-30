using Antlr4.Runtime;
using Antlr4.Runtime.Misc;
using Antlr4.Runtime.Tree;

namespace Projekt
{
    public class TypeCheckingListener : PLCBaseListener
    {
        // Tabulka symbolů pro ukládání deklarovaných proměnných
        SymbolTable symbolTable = new SymbolTable();

        // Místo výsledných čísel ukládáme datové typy (DataType) jednotlivých uzlů
        ParseTreeProperty<DataType> types = new ParseTreeProperty<DataType>();

        // Seznam chyb pro finální vypsání
        public List<string> Errors { get; } = new List<string>();

        private void AddError(ParserRuleContext context, string msg)
        {
            Errors.Add($"Line {context.Start.Line}:{context.Start.Column} - {msg}");
        }

        // --- LITERÁLY ---

        public override void ExitINT([NotNull] PLCParser.INTContext context)
        {
            types.Put(context, DataType.Int);
        }

        public override void ExitFLOAT([NotNull] PLCParser.FLOATContext context)
        {
            types.Put(context, DataType.Float);
        }

        public override void ExitBOOL([NotNull] PLCParser.BOOLContext context)
        {
            types.Put(context, DataType.Bool);
        }

        public override void ExitSTR([NotNull] PLCParser.STRContext context)
        {
            types.Put(context, DataType.String);
        }

        public override void ExitVARID([NotNull] PLCParser.VARIDContext context)
        {
            DataType type = symbolTable.GetType(context.GetText());
            if (type == DataType.Error)
            {
                AddError(context, $"Variable '{context.GetText()}' not declared.");
            }
            types.Put(context, type);
        }

        // --- VÝRAZY ---

        public override void ExitPAREN([NotNull] PLCParser.PARENContext context)
        {
            // Typ závorky je stejný jako typ výrazu uvnitř
            types.Put(context, types.Get(context.expr()));
        }

        public override void ExitADD_SUB_CONCAT([NotNull] PLCParser.ADD_SUB_CONCATContext context)
        {
            var left = types.Get(context.expr()[0]);
            var right = types.Get(context.expr()[1]);
            var op = context.op.Type;

            if (op == PLCLexer.CONCAT) // Operátor '.'
            {
                if (left == DataType.String && right == DataType.String)
                    types.Put(context, DataType.String);
                else
                {
                    AddError(context, "Operator '.' requires two strings.");
                    types.Put(context, DataType.Error);
                }
            }
            else // Operátory '+' nebo '-'
            {
                // Pravidla pro čísla (int + int = int, cokoli s floatem = float)
                if (left == DataType.Int && right == DataType.Int)
                    types.Put(context, DataType.Int);
                else if ((left == DataType.Int || left == DataType.Float) &&
                         (right == DataType.Int || right == DataType.Float))
                    types.Put(context, DataType.Float);
                else
                {
                    AddError(context, $"Cannot use operator {context.op.Text} on {left} and {right}.");
                    types.Put(context, DataType.Error);
                }
            }
        }

        public override void ExitMUL_DIV_MOD([NotNull] PLCParser.MUL_DIV_MODContext context)
        {
            var left = types.Get(context.expr()[0]);
            var right = types.Get(context.expr()[1]);
            var op = context.op.Type;

            if (op == PLCLexer.MOD) // Modulo '%' funguje jen na int
            {
                if (left == DataType.Int && right == DataType.Int)
                    types.Put(context, DataType.Int);
                else
                {
                    AddError(context, "Operator '%' requires two integers.");
                    types.Put(context, DataType.Error);
                }
            }
            else // '*' a '/'
            {
                if (left == DataType.Int && right == DataType.Int)
                    types.Put(context, DataType.Int);
                else if ((left == DataType.Int || left == DataType.Float) &&
                         (right == DataType.Int || right == DataType.Float))
                    types.Put(context, DataType.Float);
                else
                {
                    AddError(context, "Arithmetic operators require numeric types.");
                    types.Put(context, DataType.Error);
                }
            }
        }

        // --- PŘÍKAZY ---

        public override void ExitCMDVAR([NotNull] PLCParser.CMDVARContext context)
        {
            // Získání typu z klíčového slova (int, float...)
            string typeName = context.vartype().GetText();
            DataType type = typeName switch
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
                {
                    AddError(context, $"Variable '{id.GetText()}' already declared.");
                }
            }
        }

        public override void ExitASSIGN([NotNull] PLCParser.ASSIGNContext context)
        {
            DataType varType = symbolTable.GetType(context.VARID().GetText());
            DataType exprType = types.Get(context.expr());

            if (varType == DataType.Error) return;

            // Kontrola: Float do Int nelze přiřadit
            if (varType == DataType.Int && exprType == DataType.Float)
            {
                AddError(context, "Cannot assign float value to int variable.");
            }
            // Zbytek musí sedět nebo být int -> float (automatický cast)
            else if (varType != exprType && !(varType == DataType.Float && exprType == DataType.Int))
            {
                AddError(context, $"Incompatible types: {varType} and {exprType}.");
            }

            types.Put(context, varType);
        }
    }
}