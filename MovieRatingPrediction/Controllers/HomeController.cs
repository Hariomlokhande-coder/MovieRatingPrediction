using Microsoft.AspNetCore.Mvc;
using MovieRatingPrediction.Models;
using MovieRatingPrediction.Services;
using System.Diagnostics;

namespace MovieRatingPrediction.Controllers
{
    public class HomeController : Controller
    {
        private readonly ILogger<HomeController> _logger;
        private readonly MovieMLService _mlService;

        public HomeController(ILogger<HomeController> logger, MovieMLService mlService)
        {
            _logger = logger;
            _mlService = mlService;
        }

        public IActionResult Index()
        {
            ViewBag.RMSE = Math.Round(_mlService.RMSE, 4);
            ViewBag.MAE = Math.Round(_mlService.MAE, 4);
            ViewBag.R2 = Math.Round(_mlService.RSquared, 4);
            return View();
        }

        [HttpPost]
        public IActionResult Predict(string genre, float? duration, float? budget, float? popularity)
        {
            try
            {
                if (string.IsNullOrEmpty(genre) || duration == null || budget == null || popularity == null)
                {
                    ViewBag.Error = "All fields are required.";
                    return View("Index");
                }

                var input = new MovieData
                {
                    Genre = genre,
                    Duration = duration.Value,
                    Budget = budget.Value,
                    Popularity = popularity.Value
                };

                var result = _mlService.Predict(input);

                ViewBag.Genre = genre;
                ViewBag.Duration = duration;
                ViewBag.Budget = budget;
                ViewBag.Popularity = popularity;
                
                ViewBag.RMSE = Math.Round(_mlService.RMSE, 4);
                ViewBag.MAE = Math.Round(_mlService.MAE, 4);
                ViewBag.R2 = Math.Round(_mlService.RSquared, 4);
                ViewBag.Result = Math.Round(result, 2);
                return View("Index");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Prediction failed");
                ViewBag.Error = "Something went wrong.";
                return View("Index");
            }
        }

        public IActionResult Privacy()
        {
            return View();
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }
    }
}