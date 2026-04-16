using Antlr4.Runtime;
using Antlr4.Runtime.Misc;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Projekt
{
    public class VerboseListener : BaseErrorListener
    {
        public override void SyntaxError( System.IO.TextWriter output, IRecognizer recognizer, IToken offendingSymbol,int line, int charPositionInLine,string msg, RecognitionException e)
        {
            IList<string> stack = ((Parser)recognizer).GetRuleInvocationStack();
            stack.Reverse();

            output.WriteLine("rule stack: " + String.Join(", ", stack));
            output.WriteLine("line " + line + ":" + charPositionInLine + " at " + offendingSymbol + ": " + msg);
        }
    }
}
