namespace GPURepair.Repair.Metadata
{
    using System;
    using Microsoft.Boogie;

    public class Barrier
    {
        /// <summary>
        /// The constructor that takes a name.
        /// </summary>
        /// <param name="name">The barrier name.</param>
        public Barrier(string name)
        {
            Name = name;
        }

        /// <summary>
        /// The weight associated with the loop-depth.
        /// </summary>
        public static int LoopDepthWeight { get; set; }

        /// <summary>
        /// The weight associated with a grid-level barrier.
        /// </summary>
        public static int GridBarrierWeight { get; set; }

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
        /// The call command corresponding to the barrier.
        /// </summary>
        public CallCmd Call { get; set; }

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
                int loopWeight = Convert.ToInt32(Math.Pow(LoopDepthWeight, LoopDepth));
                int barrierWeight = GridLevel ? GridBarrierWeight : 0;

                return loopWeight + barrierWeight;
            }
        }
    }
}
