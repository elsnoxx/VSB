using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Ukol4
{
    public class AgeComparer : IComparer<Customer>
    {
        public int Compare(Customer x, Customer y)
        {
            if (x == null || y == null)
            {
                throw new ArgumentException("Customers cannot be null.");
            }

            return x.Age.CompareTo(y.Age);
        }
    }

}
