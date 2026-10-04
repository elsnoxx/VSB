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
        public bool HadError { get; private set; } = false;

        public override void SyntaxError(IRecognizer recognizer, IToken offendingSymbol, int line, int charPositionInLine, string msg, RecognitionException e)
        {
            HadError = true;
            Console.Error.WriteLine($"Syntax Error: line {line}:{charPositionInLine} {msg}");
        }
    }
}
