namespace GPURepair.Repair.Metadata
{
    using System.IO;
    using GPUVerify;

    public class Location
    {
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
            FileInfo file = new FileInfo(File);
            return string.Format("{0} {1}", file.FullName, Line);
        }

        /// <summary>
        /// Determines whether the specified object is equal to the current object.
        /// </summary>
        /// <param name="obj">The object to compare with the current object.</param>
        /// <returns>true if the specified object is equal to the current object; otherwise, false.</returns>
        public override bool Equals(object obj)
        {
            if (obj is Location)
            {
                Location location = obj as Location;
                return Directory == location.Directory && File == location.File && Line == location.Line && Column == location.Column;
            }
            else if (obj is SourceLocationInfo.Record)
            {
                SourceLocationInfo.Record source = obj as SourceLocationInfo.Record;
                return source.GetDirectory() == Directory && source.GetFile() == File
                    && source.GetLine() == Line && source.GetColumn() == Column;
            }

            return false;
        }

        /// <summary>
        /// Serves as the default hash function.
        /// </summary>
        /// <returns>A hash code for the current object.</returns>
        public override int GetHashCode()
        {
            return base.GetHashCode();
        }

        /// <summary>
        /// Determines if the current location is between the given two locations.
        /// </summary>
        /// <param name="first">The first location.</param>
        /// <param name="second">The second location.</param>
        /// <returns>true if the current location is between the given two locations; otherwise, false.</returns>
        public bool IsBetween(Location first, Location second)
        {
            if (Directory != first.Directory || File != first.File)
                return false;
            if (Directory != second.Directory || File != second.File)
                return false;

            if (Line == first.Line && Line == second.Line)
                // Boogies evaluates the right-side operation before the left-side
                return first.Column >= Column && Column >= second.Column;
            return second.Line >= Line && Line > first.Line;
        }
    }
}
