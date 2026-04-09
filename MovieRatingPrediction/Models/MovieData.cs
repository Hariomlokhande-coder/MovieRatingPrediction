using Microsoft.ML.Data;

namespace MovieRatingPrediction.Models
{
    public class MovieData
    {
        [LoadColumn(0)]
        public string Genre { get; set; }

        [LoadColumn(1)]
        public float Duration { get; set; }

        [LoadColumn(2)]
        public float Budget { get; set; }

        [LoadColumn(3)]
        public float Popularity { get; set; }

        [LoadColumn(4)]
        public float Rating { get; set; }
    }
}
