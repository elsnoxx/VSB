using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Ukol7
{
    internal class TreeMap<TKey, TValue>
    {
        private Node<TKey, TValue>? root;
        private int count;
        public int Count
        {
            get { return count; }
        }

        public Node<TKey, TValue> Get(TKey key) 
        {
            var current = root;
            while (current != null) 
            { 
                int compare = key.Com
            }
            return null;
        
        }
        public bool Set(Node<TKey, TValue> node) { return false; }

        public Node<TKey, TValue> this[TKey keyIndex]
        {
            get
            {
                return this.Get(keyIndex);
            }
            set
            {
                this.Set(value);
            }
        }

    }
}
