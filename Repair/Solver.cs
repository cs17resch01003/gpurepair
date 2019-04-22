using System.Collections.Generic;
using System.Linq;

namespace GPURepair.Repair
{
    public class Solver
    {
        /// <summary>
        /// Determine the barrier assignments based on the error traces.
        /// </summary>
        /// <param name="errors">The errors.</param>
        /// <returns>The barrier assignments.</returns>
        public Dictionary<string, bool> Solve(List<Error> errors)
        {
            // Use Z3 and figure out the variable assignments
            Dictionary<string, bool> assignments = new Dictionary<string, bool>();
            if (errors.Any())
                assignments.Add("b1", true);

            return assignments;
        }
    }
}