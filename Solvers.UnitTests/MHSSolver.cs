using GPURepair.Solvers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.IO;

namespace Solvers.UnitTests
{
    [TestClass]
    public class MHSSolver
    {
        [TestMethod]
        public void RunTestSuite()
        {
            string[] files = Directory.GetFiles(@"TestSuite");
            foreach (string file in files)
                ProcessFile(file);
        }

        private void ProcessFile(string filePath)
        {
            List<Clause> clauses = GetClauses(filePath);

            GPURepair.Solvers.MHSSolver solver = new GPURepair.Solvers.MHSSolver(clauses);
            Dictionary<string, bool> solution = solver.Solve();

            Verify(clauses, solution, filePath);
        }

        private List<Clause> GetClauses(string filePath)
        {
            List<Clause> clauses = new List<Clause>();

            string[] lines = File.ReadAllLines(filePath);
            foreach (string line in lines)
            {
                if (line.StartsWith("c ", StringComparison.InvariantCultureIgnoreCase))
                    continue;
                else if (line.StartsWith("p ", StringComparison.InvariantCultureIgnoreCase))
                {
                    if (line.StartsWith("p cnf ", StringComparison.InvariantCultureIgnoreCase))
                        continue;
                    else
                        throw new Exception("Only cnf is supported!");
                }
                else
                {
                    Clause clause = new Clause();
                    string[] variables = line.Split(' ');

                    foreach (string variable in variables)
                    {
                        int literal = int.Parse(variable);

                        if (literal > 0)
                            clause.Add(new Literal(literal.ToString(), true));
                        else if (literal < 0)
                            clause.Add(new Literal((-literal).ToString(), false));
                    }

                    clauses.Add(clause);
                }
            }

            return clauses;
        }

        private void Verify(List<Clause> clauses,
            Dictionary<string, bool> solution, string filePath)
        {
            foreach (Clause clause in clauses)
            {
                bool sat = false;
                foreach (Literal literal in clause.Literals)
                    if (solution.ContainsKey(literal.Variable) && literal.Value == solution[literal.Variable])
                    {
                        sat = true;
                        break;
                    }

                if (sat == false)
                    Assert.Fail(filePath);
            }
        }
    }
}