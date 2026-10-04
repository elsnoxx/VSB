using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace Cviko1
{
    public class Test
    {
        public static void testSQL()
        {
            Assembly a = Assembly.GetExecutingAssembly();
            Console.WriteLine(a);

            Person p = new Person();
            p.Name = "John";
            p.email = "john@john.cz";

            string sql = "INSERT INTO Person (";

            foreach (var prop in typeof(Person).GetProperties())
            {
                var attr = prop.GetCustomAttributes(typeof(ToSQLAttribute), false).FirstOrDefault() as ToSQLAttribute;
                if (attr != null)
                {
                    sql += attr.ColumnName + ", ";
                }
            }

            sql += ") VALUES (";

            foreach (var prop in typeof(Person).GetProperties())
            {
                var attr = prop.GetCustomAttributes(typeof(ToSQLAttribute), false).FirstOrDefault() as ToSQLAttribute;
                if (attr != null)
                {
                    sql += "'" + prop.GetValue(p) + "', ";
                }
            }

            sql += ");";

            Console.WriteLine(sql);
        }
    }
}
