using Microsoft.Boogie;

namespace GPURepair.Repair.Metadata
{
    /// <summary>
    /// Metadata related to a barrier.
    /// </summary>
    public class Barrier
    {
        /// <summary>
        /// The name of the barrier.
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// The implementation corresponding to the barrier.
        /// </summary>
        public Implementation Implementation { get; set; }

        /// <summary>
        /// The block corresponding to the barrier.
        /// </summary>
        public Block Block { get; set; }

        /// <summary>
        /// The varibale corresponding to the barrier.
        /// </summary>
        public Variable Variable { get; set; }

        /// <summary>
        /// The location of the barrier.
        /// </summary>
        public int SourceLocation { get; set; }

        /// <summary>
        /// Indicates if the barrier was generated.
        /// </summary>
        public bool Generated { get; set; }
    }
}
