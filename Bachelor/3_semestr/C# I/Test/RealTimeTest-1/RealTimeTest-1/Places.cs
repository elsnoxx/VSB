using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RealTimeTest_1
{
    public class Places
    {

        public List<Places> _items; 

        //public void Add (T place)
        //{
        //    _items.Add(place);
        //}
            
        

        public void Sort()
        {
            foreach (var item in _items)
            {
                
            }
        }

        public void Save(string filename)
        {
            using StreamWriter streamWriter = new StreamWriter(filename);
            {
                foreach (var item in _items)
                {
                    streamWriter.WriteLine(item);
                }
                streamWriter.Close();
                streamWriter.Flush();
            }
        }
    }
}
