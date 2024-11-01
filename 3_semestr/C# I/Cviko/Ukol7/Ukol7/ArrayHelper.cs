using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Ukol7
{
    internal class ArrayHelper
    {
        public static void Swap<T>(T[] array, uint index1, uint index2)
        {
            if (array == null || index1 >= array.Length || index2 >= array.Length )
            {
                //throw new ArgumentException("Bad inputs");
                return;
            }


            T temp = array[index1];
            array[index1] = array[index2];
            array[index2] = temp;
        }

        public static T[] Concat<T>(T[] array1, T[] array2)
        {
            if (array1 == null || array2 == null)
            {
                //throw new ArgumentException("Empty inputs");
                return Array.Empty<T>();
            }


            T[] result = new T[array1.Length + array2.Length];
            array1.CopyTo(result, 0);
            array2.CopyTo(result, array1.Length);
            return result;
        }
    }
}
