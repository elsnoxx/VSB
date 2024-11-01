using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Ukol7
{
    public interface IKeyValuePair<TKey, TValue>
    {
        TKey Key { get; }
        TValue Value { get; }
    }
}
