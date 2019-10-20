namespace GPURepair.Repair.Metadata
{
    /// <summary>
    /// Metadata related to a location.
    /// </summary>
    public class Location
    {
        /// <summary>
        /// The source location.
        /// </summary>
        public int SourceLocation { get; set; }

        /// <summary>
        /// The line in the source file.
        /// </summary>
        public int Line { get; set; }

        /// <summary>
        /// The column in the source file.
        /// </summary>
        public int Column { get; set; }

        /// <summary>
        /// The source file.
        /// </summary>
        public string File { get; set; }

        /// <summary>
        /// The source directory.
        /// </summary>
        public string Directory { get; set; }

        /// <summary>
        /// Converts the object to a string.
        /// </summary>
        /// <returns>The string representation.</returns>
        public override string ToString()
        {
            return string.Format("{0} {1}", File, Line);
        }
    }
}
