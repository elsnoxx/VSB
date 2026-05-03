using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

namespace Projekt
{
    public class VirtualMachine
    {
        private Stack<object> stack = new Stack<object>();
        private List<string[]> instructions = new List<string[]>();
        private Dictionary<string, object> memory = new Dictionary<string, object>();
        private Dictionary<string, int> labels = new Dictionary<string, int>();

        public VirtualMachine(string generatedCode)
        {
            var lines = generatedCode.Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
            int index = 0;

            foreach (var line in lines)
            {
                var trimmed = line.Trim();

                if (trimmed.StartsWith("label"))
                {
                    labels[trimmed.Split(' ')[1]] = index;
                }
                else
                {
                    instructions.Add(trimmed.Split(' ', StringSplitOptions.RemoveEmptyEntries));
                    index++;
                }
            }
        }

        public void Run()
        {
            int ip = 0;

            while (ip < instructions.Count)
            {
                var inst = instructions[ip];
                string cmd = inst[0].ToLower();

                //Console.WriteLine($"DEBUG: IP={ip}, Inst={string.Join(" ", inst)}, StackCount={stack.Count}");

                switch (cmd)
                {
                    case "push":
                        Push(inst);
                        break;

                    case "pop":
                        stack.Pop();
                        break;

                    case "load":
                        if (!memory.ContainsKey(inst[1]))
                            memory[inst[1]] = 0;

                        stack.Push(memory[inst[1]]);
                        break;

                    case "save":
                        var val = stack.Pop();

                        //if (!memory.ContainsKey(inst[1]))
                        //    memory[inst[1]] = val;

                        memory[inst[1]] = val;
                        break;

                    case "print":
                        Print(inst);
                        break;

                    case "read":
                        Read(inst);
                        break;

                    case "add":
                        ApplyBinary(inst[1], "add");
                        break;

                    case "sub":
                        ApplyBinary(inst[1], "sub");
                        break;

                    case "mul":
                        ApplyBinary(inst[1], "mul");
                        break;

                    case "div":
                        ApplyBinary(inst[1], "div");
                        break;

                    case "mod":
                        int right = (int)stack.Pop();
                        int left = (int)stack.Pop();
                        stack.Push(left % right);
                        break;

                    case "uminus":
                        ApplyUnary(inst[1], a => -a);
                        break;

                    case "concat":
                        var s2 = stack.Pop().ToString();
                        var s1 = stack.Pop().ToString();
                        stack.Push(s1 + s2);
                        break;

                    case "and":
                        stack.Push((bool)stack.Pop() & (bool)stack.Pop());
                        break;

                    case "or":
                        stack.Push((bool)stack.Pop() | (bool)stack.Pop());
                        break;

                    case "not":
                        stack.Push(!(bool)stack.Pop());
                        break;

                    case "gt":
                        ApplyBinary(inst[1], "gt");
                        break;

                    case "lt":
                        ApplyBinary(inst[1], "lt");
                        break;

                    case "eq":
                        string eqType = inst[1];
                        var val2 = stack.Pop();
                        var val1 = stack.Pop();
                        stack.Push(val1.Equals(val2));
                        break;

                    case "itof":
                        int intVal = (int)stack.Pop();
                        stack.Push((float)intVal);
                        break;

                    case "jmp":
                        ip = labels[inst[1]];
                        continue;

                    case "fjmp":
                        var r = stack.Pop();
                        bool cond = r switch
                        {
                            bool b => b,
                            int i => i != 0,
                            _ => throw new Exception("Expected bool/int")
                        };
                        if (!cond)
                        {
                            ip = labels[inst[1]];
                            continue;
                        }
                        break;
                }

                ip++;
            }
        }

        private void Push(string[] inst)
        {
            string type = inst[1];
            string value = string.Join(" ", inst.Skip(2));

            switch (type)
            {
                case "I":
                    stack.Push(int.Parse(value));
                    break;
                case "F":
                    stack.Push(float.Parse(value));
                    break;
                case "B":
                    stack.Push(bool.Parse(value));
                    break;
                case "S":
                    stack.Push(value.Trim('"'));
                    break;
            }
        }

        private void Read(string[] inst)
        {
            string type = inst[1];
            Console.Write($"\n[VM WAITING FOR INPUT ({type})]: ");
            string input = Console.ReadLine();

            if (string.IsNullOrEmpty(input)) input = "0";

            switch (type)
            {
                case "I":
                    stack.Push(int.Parse(input));
                    break;
                case "F":
                    stack.Push(float.Parse(input, CultureInfo.InvariantCulture));
                    break;
                case "B":
                    stack.Push(bool.Parse(input));
                    break;
                case "S":
                    stack.Push(input);
                    break;
            }
        }

        private void Print(string[] inst)
        {
            int count = int.Parse(inst[1]);
            var values = new List<object>();

            for (int i = 0; i < count; i++)
                values.Add(stack.Pop());

            values.Reverse();
            Console.WriteLine(string.Join(" ", values));
        }

        private void ApplyBinary(string type, string opName)
        {
            var b = stack.Pop();
            var a = stack.Pop();

            if (type == "I")
            {
                int valA = Convert.ToInt32(a);
                int valB = Convert.ToInt32(b);
                stack.Push(opName switch
                {
                    "add" => valA + valB,
                    "sub" => valA - valB,
                    "mul" => valA * valB,
                    "div" => valA / valB,
                    "gt" => valA > valB,
                    "lt" => valA < valB,
                    _ => throw new Exception($"Unknown int op: {opName}")
                });
            }
            else if (type == "F")
            {
                float valA = Convert.ToSingle(a);
                float valB = Convert.ToSingle(b);
                stack.Push(opName switch
                {
                    "add" => valA + valB,
                    "sub" => valA - valB,
                    "mul" => valA * valB,
                    "div" => valA / valB,
                    "gt" => valA > valB,
                    "lt" => valA < valB,
                    _ => throw new Exception($"Unknown float op: {opName}")
                });
            }
        }

        private void ApplyUnary(string type, Func<dynamic, dynamic> op)
        {
            dynamic a = stack.Pop();
            stack.Push(op(a));
        }
    }
}