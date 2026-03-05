using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Lab01.tree
{
    public class BinaryOperator : Node
    {
        public char Operator { get; set; }
        public Node Left { get; set; }
        public Node Right { get; set; }

        public BinaryOperator(char op, Node left, Node right)
        {
            Operator = op;
            Left = left;
            Right = right;
        }
    }
}
