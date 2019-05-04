using GPURepair.Solvers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.IO;

namespace Solvers.UnitTests
{
    [TestClass]
    public class MaxSATSolver
    {
        [TestMethod]
        public void RunTestSuite()
        {
            string[] files = Directory.GetFiles(@"TestSuite\MaxSATSolver");
            foreach (string file in files)
                ProcessFile(file);
        }

        private void ProcessFile(string filePath)
        {
            dynamic clauses = GetClauses(filePath);

            GPURepair.Solvers.MaxSATSolver solver = new GPURepair.Solvers.MaxSATSolver(clauses.HardClauses, clauses.SoftClauses);
            Dictionary<string, bool> solution = solver.Solve();

            Verify(clauses.HardClauses, solution, filePath);
        }

        private dynamic GetClauses(string filePath)
        {
            List<Clause> hard_clauses = new List<Clause>();
            List<Clause> soft_clauses = new List<Clause>();

            int? hard_clauses_count = null;

            string[] lines = File.ReadAllLines(filePath);
            foreach (string line in lines)
            {
                string[] variables = line.Split(' ');
                if (hard_clauses_count == null)
                {
                    hard_clauses_count = int.Parse(variables[0]);
                    continue;
                }

                Clause clause = new Clause();
                foreach (string variable in variables)
                {
                    if (!variable.StartsWith("-"))
                        clause.Add(new Literal(variable, true));
                    else
                        clause.Add(new Literal(variable.Replace("-", string.Empty), false));
                }

                if (hard_clauses_count != 0)
                {
                    hard_clauses.Add(clause);
                    hard_clauses_count--;
                }
                else
                {
                    soft_clauses.Add(clause);
                }
            }

            return new
            {
                HardClauses = hard_clauses,
                SoftClauses = soft_clauses
            };
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