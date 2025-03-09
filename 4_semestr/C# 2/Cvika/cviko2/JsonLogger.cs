using System.ComponentModel;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace cviko2
{
    public class JsonLogger : IMyLogger
    {
        public JsonLogger() { }

        public void Log(string message)
        {
            // writing data to the json file
            //File.WriteAllText("log.json", message + '\n');
            string path = "log.json";
            List<string> logs = new List<string>();

            if (File.Exists(path))
            {
                logs = JsonSerializer.Deserialize<List<string>>(File.ReadAllText(path));
            }
            else
            {
                logs = new List<string>();
            }

            logs.Add(message);

            File.WriteAllText(path, JsonSerializer.Serialize(logs));

            Console.WriteLine("JsonLogger: " + message);
        }
    }
}
