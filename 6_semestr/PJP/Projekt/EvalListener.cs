using Antlr4.Runtime.Misc;
using Antlr4.Runtime.Tree;
using PLCProject;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Projekt
{
    public class EvalListener : PLCBaseListener
    {
        ParseTreeProperty<int> values = new ParseTreeProperty<int>();

        public override void ExitInt([NotNull] PLCParser.IntContext context)
        {
            values.Put(context, Convert.ToInt32(context.INT().GetText(), 10));
        }
        public override void ExitHexa([NotNull] PLCParser.HexaContext context)
        {
            values.Put(context, Convert.ToInt32(context.HEXA().GetText(), 16));
        }

        public override void ExitOct([NotNull] PLCParser.OctContext context)
        {
            values.Put(context, Convert.ToInt32(context.OCT().GetText(), 8));
        }
        public override void ExitPar([NotNull] PLCParser.ParContext context)
        {
            values.Put(context, values.Get(context.expr()));
        }
        public override void ExitAdd([NotNull] PLCParser.AddContext context)
        {
            var left = values.Get(context.expr()[0]);
            var right = values.Get(context.expr()[1]);
            if (context.op.Text.Equals("+"))
            {
                values.Put(context, left + right);
            }
            else
            {
                values.Put(context, left - right);
            }
        }
        public override void ExitMul([NotNull] PLCParser.MulContext context)
        {
            var left = values.Get(context.expr()[0]);
            var right = values.Get(context.expr()[1]);
            if (context.op.Text.Equals("*"))
            {
                values.Put(context, left * right);
            }
            else
            {
                values.Put(context, left / right);
            }
        }
        public override void ExitProg([NotNull] PLCParser.ProgContext context)
        {
            foreach (var expr in context.expr())
            {
                Console.WriteLine(values.Get(expr));
            }
        }
    }
}
