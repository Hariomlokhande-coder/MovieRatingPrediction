using Microsoft.ML;
using Microsoft.ML.Data;
using MovieRatingPrediction.Models;
using System;
using System.Collections.Generic;

namespace MovieRatingPrediction.Services
{
    public class MovieMLService
    {
        private readonly MLContext _mlContext;
        private ITransformer _model;
        private PredictionEngine<MovieData, MoviePrediction> _predictionEngine;

        public double RMSE { get; private set; }
        public double MAE { get; private set; }
        public double RSquared { get; private set; }

        public MovieMLService()
        {
            _mlContext = new MLContext(seed: 42);
            TrainModel();
        }

        private void TrainModel()
        {
            var watch = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                // Load data and immediately Shuffle to prevent bias
                var data = _mlContext.Data.LoadFromTextFile<MovieData>(
                    path: "Data/movies.csv",
                    hasHeader: true,
                    separatorChar: ',');

                if (data == null)
                    throw new Exception("Dataset is null");

                // Data Cleaning: Remove rows with invalid values (Negative budget/duration)
                // This ensures model doesn't learn from noise
                data = _mlContext.Data.FilterRowsByColumn(data, "Duration", lowerBound: 1, upperBound: 500); // Movies between 1 and 500 mins
                data = _mlContext.Data.FilterRowsByColumn(data, "Budget", lowerBound: 0); // Non-negative budget
                data = _mlContext.Data.FilterRowsByColumn(data, "Rating", lowerBound: 0, upperBound: 10); // Rating usually 0-10

                // Shuffle for better generalization
                data = _mlContext.Data.ShuffleRows(data, seed: 42);

                // Improved Data Processing Pipeline (Optimized for High R2)
                var dataProcessPipeline = _mlContext.Transforms
                    .ReplaceMissingValues(new[]
                    {
                        new InputOutputColumnPair("Duration"),
                        new InputOutputColumnPair("Budget"),
                        new InputOutputColumnPair("Popularity")
                    })
                    // Handling categorical data
                    .Append(_mlContext.Transforms.Categorical.OneHotEncoding("GenreEncoded", "Genre"))
                    // Feature Engineering: Log Budget & Normalization
                    .Append(_mlContext.Transforms.NormalizeLogMeanVariance("BudgetLog", "Budget"))
                    .Append(_mlContext.Transforms.NormalizeMeanVariance("DurationNorm", "Duration"))
                    .Append(_mlContext.Transforms.NormalizeMeanVariance("PopularityNorm", "Popularity"))
                    // ADVANCED: Budget per Minute (Captures production efficiency)
                    .Append(_mlContext.Transforms.Expression("BudgetPerMinute", "(Budget, Duration) => Budget / Duration", "Budget", "Duration"))
                    // ADVANCED: Popularity Squared (Captures non-linear impact of fame)
                    .Append(_mlContext.Transforms.Expression("PopularitySquared", "(p) => p * p", "Popularity"))
                    // Combine all features (Original + Engineered)
                    .Append(_mlContext.Transforms.Concatenate("Features",
                        "DurationNorm", "BudgetLog", "PopularityNorm", "GenreEncoded", "BudgetPerMinute", "PopularitySquared"));


                var dataCount = _mlContext.Data.CreateEnumerable<MovieData>(data, reuseRowObject: false);
                if (dataCount == null)
                    throw new Exception("Data enumeration failed");

                // Train/Test Split (VERY IMPORTANT)
                var splitData = _mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

                if (splitData.TrainSet == null || splitData.TestSet == null)
                    throw new Exception("Train/Test split failed");

                // Multiple Model Training (Optimized for 300k+ rows)
                var trainers = new Dictionary<string, IEstimator<ITransformer>>
                {
                    { "LightGBM", _mlContext.Regression.Trainers.LightGbm(new Microsoft.ML.Trainers.LightGbm.LightGbmRegressionTrainer.Options
                        {
                            LabelColumnName = "Rating",
                            FeatureColumnName = "Features",
                            NumberOfLeaves = 128,
                            MinimumExampleCountPerLeaf = 30,
                            NumberOfIterations = 500,
                            LearningRate = 0.05
                        }) 
                    },
                    { "FastTree", _mlContext.Regression.Trainers.FastTree(new Microsoft.ML.Trainers.FastTree.FastTreeRegressionTrainer.Options
                        {
                            LabelColumnName = "Rating",
                            FeatureColumnName = "Features",
                            NumberOfLeaves = 70,
                            NumberOfTrees = 300,
                            LearningRate = 0.1
                        }) 
                    },
                    { "FastForest", _mlContext.Regression.Trainers.FastForest(new Microsoft.ML.Trainers.FastTree.FastForestRegressionTrainer.Options
                        {
                            LabelColumnName = "Rating",
                            FeatureColumnName = "Features",
                            NumberOfLeaves = 50,
                            NumberOfTrees = 200
                        }) 
                    },
                    { "SDCA", _mlContext.Regression.Trainers.Sdca(new Microsoft.ML.Trainers.SdcaRegressionTrainer.Options
                        {
                            LabelColumnName = "Rating",
                            FeatureColumnName = "Features",
                            MaximumNumberOfIterations = 100
                        }) 
                    }
                };

                double bestRSquared = double.MinValue;
                ITransformer bestModel = null;

                foreach (var item in trainers)
                {
                    var trainerWatch = System.Diagnostics.Stopwatch.StartNew();

                    var model = dataProcessPipeline
                         .Append(item.Value)
                         .Fit(splitData.TrainSet);

                    trainerWatch.Stop();

                    var predictions = model.Transform(splitData.TestSet);
                    var metrics = _mlContext.Regression.Evaluate(predictions, labelColumnName: "Rating");

                    // LOGGING COMPARISON
                    Console.WriteLine($"\n--- Trainer: {item.Key} ---");
                    Console.WriteLine($"   Execution Time:        {trainerWatch.ElapsedMilliseconds} ms");
                    Console.WriteLine($"   R2 Score (Smartness):  {metrics.RSquared:F4}");
                    Console.WriteLine($"   RMSE (Error Distance): {metrics.RootMeanSquaredError:F4}");
                    Console.WriteLine($"   MAE (Average Error):   {metrics.MeanAbsoluteError:F4}");

                    // SMART SELECTION: We pick the model with the highest R2 Score
                    if (metrics.RSquared > bestRSquared)
                    {
                        bestRSquared = metrics.RSquared;
                        bestModel = model;

                        this.RMSE = metrics.RootMeanSquaredError;
                        this.MAE = metrics.MeanAbsoluteError;
                        this.RSquared = metrics.RSquared;
                    }
                }

                _model = bestModel;

                Console.WriteLine($"\n >> SUCCESS: Best Model Selected with R2 Score: {this.RSquared:F4}");

                if (_model == null)
                    throw new Exception("Model training failed");

                //Transform()
                var predictionsFinal = _model.Transform(splitData.TestSet);

                if (predictionsFinal == null)
                    throw new Exception("Prediction transformation failed");

                if (this.RSquared < 0)
                    Console.WriteLine("⚠ Warning: Model performance is poor (R2 < 0)");

                //create Prediction Engine
                _predictionEngine = _mlContext.Model
                    .CreatePredictionEngine<MovieData, MoviePrediction>(_model);

                if (_predictionEngine == null)
                    throw new Exception("Prediction engine creation failed");

                // ✅ Save Model
                _mlContext.Model.Save(_model, data.Schema, "Model/movie_model.zip");

                watch.Stop();
                Console.WriteLine($"\n >> TOTAL TRAINING TIME: {watch.Elapsed.TotalSeconds:F2} seconds");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error during training: {ex.Message}");
                throw;
            }
        }

        public float Predict(MovieData input)
        {
            try
            {

                if (input == null)
                    throw new ArgumentNullException(nameof(input));

                if (input.Duration <= 0 || input.Budget < 0)
                    throw new ArgumentException("Invalid input data: Duration must be > 0 and Budget must be >= 0");

                //Edge Case: Model not loaded
                if (_predictionEngine == null)
                    throw new Exception("Model is not initialized");

                var result = _predictionEngine.Predict(input);

                //  Invalid prediction
                if (float.IsNaN(result.PredictedRating) || float.IsInfinity(result.PredictedRating))
                    throw new Exception("Invalid prediction result");

                return result.PredictedRating;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Prediction error: {ex.Message}");
                throw;
            }
        }
    }
}