﻿using GPURepair.Solvers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
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
            string[] files = Directory.GetFiles(@"TestSuite\MHSSolver");
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
                Clause clause = new Clause();

                string[] variables = line.Split(' ');
                foreach (string variable in variables)
                {
                    if (!variable.StartsWith("-"))
                        clause.Add(new Literal(variable, true));
                    else
                        clause.Add(new Literal(variable.Replace("-", string.Empty), false));
                }

                clauses.Add(clause);
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