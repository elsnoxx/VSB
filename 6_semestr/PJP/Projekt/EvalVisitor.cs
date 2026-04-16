using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Antlr4.Runtime;
using Antlr4.Runtime.Misc;
using PLCProject;

namespace Projekt
{
    public class EvalVisitor : PLCBaseVisitor<int>
    {
        public override int VisitInt([NotNull] PLCParser.IntContext context)
        {
            return Convert.ToInt32(context.INT().GetText(), 10);
        }
        public override int VisitHexa([NotNull] PLCParser.HexaContext context)
        {
            return Convert.ToInt32(context.HEXA().GetText(), 16);
        }
        public override int VisitOct([NotNull] PLCParser.OctContext context)
        {
            return Convert.ToInt32(context.OCT().GetText(), 8);
        }
        public override int VisitPar([NotNull] PLCParser.ParContext context)
        {
            return Visit(context.expr());
        }
        public override int VisitAdd([NotNull] PLCParser.AddContext context)
        {
            var left = Visit(context.expr()[0]);
            var right = Visit(context.expr()[1]);
            if (context.op.Text.Equals("+"))
            {
                return left + right;
            }
            else
            {
                return left - right;
            }
        }
        public override int VisitMul([NotNull] PLCParser.MulContext context)
        {
            var left = Visit(context.expr()[0]);
            var right = Visit(context.expr()[1]);
            if (context.op.Text.Equals("*"))
            {
                return left * right;
            }
            else
            {
                return left / right;
            }
        }
        public override int VisitProg([NotNull] PLCParser.ProgContext context)
        {
            foreach (var expr in context.expr())
            {
                Console.WriteLine(Visit(expr));
            }
            return 0;
        }

    }
}
