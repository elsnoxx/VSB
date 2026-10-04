using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Ukol7
{
    internal class Node<Tkey, TValue> : IKeyValuePair<Tkey, TValue>
    {
        public Tkey Key { get; private set; }
        public TValue Value { get; set; }

        public Node<Tkey, TValue>? Left { get; set; }
        public Node<Tkey, TValue>? Right { get; set; }

        public Node(Tkey key, TValue value)
        {
            Key = key;
            Value = value;
        }
    }
}
