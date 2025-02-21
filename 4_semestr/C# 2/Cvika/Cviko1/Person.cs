using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Cviko1
{
    public class Person
    {
        public int ID { get; set; }

        [ToSQL("name")]
        public string Name { get; set; }
        [ToSQL]
        public string email { get; set; }

        public Person() { }
    }

    [AttributeUsage(AttributeTargets.Property)]
    public class ToSQLAttribute : Attribute
    {
        public string ColumnName { get; set; }
        public ToSQLAttribute() { }

        public ToSQLAttribute(string columnName)
        {
            ColumnName = columnName;
        }
    }
    
}
