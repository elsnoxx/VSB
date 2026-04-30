using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Projekt
{
    public class SymbolTable
    {
        private Dictionary<string, DataType> table = new Dictionary<string, DataType>();

        public bool Declare(string name, DataType type)
        {
            if (table.ContainsKey(name)) return false;
            table.Add(name, type);
            return true;
        }

        public DataType GetType(string name)
        {
            if (table.TryGetValue(name, out var type)) return type;
            return DataType.Error;
        }
    }
}
