namespace GPURepair.Repair.Metadata
{
    using System;
    using Microsoft.Boogie;

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

        /// <summary>
        /// Indicates if the barrier is a grid-level barrier.
        /// </summary>
        public bool GridLevel { get; set; }

        /// <summary>
        /// Represents the loop depth of the barrier.
        /// </summary>
        public int LoopDepth { get; set; } = 0;

        /// <summary>
        /// The weight of the barrier.
        /// </summary>
        public int Weight
        {
            get
            {
                int loopWeight = (int)(LoopDepth == 0 ? 0 : Math.Pow(2, LoopDepth));
                int barrierWeight = GridLevel ? 16 : 1;

                return loopWeight + barrierWeight;
            }
        }
    }
}
