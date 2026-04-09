using Microsoft.ML.Data;

namespace MovieRatingPrediction.Models
{
    public class MoviePrediction
    {
        [ColumnName("Score")]
        public float PredictedRating { get; set; }
    }
}
