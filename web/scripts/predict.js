// BLOCK PAGE FROM LOADING UNTIL CHECK IS DONE
if (!sessionStorage.getItem("currentUser")) {
    window.location.href = "pricing.html";
} else {
    let users = JSON.parse(sessionStorage.getItem("users")) || {};
    let currentUser = sessionStorage.getItem("currentUser");

    if (!users[currentUser]?.subscribed) {
        alert("You need a subscription to view this page.");
        window.location.href = "pricing.html";
    } else {
        document.addEventListener("DOMContentLoaded", () => {
            const homeSelect = document.getElementById("home-team");
            const awaySelect = document.getElementById("away-team");
            const predictionResult = document.getElementById("prediction-result");
            const getPredictionBtn = document.getElementById("get-prediction");

            const teams = [
                "ANA", "ARI", "BOS", "BUF", "CGY", "CAR", "CHI", "COL", "CBJ", "DAL",
                "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NSH", "NJD", "NYI", "NYR",
                "OTT", "PHI", "PIT", "SEA", "SJS", "STL", "TBL", "TOR", "VAN", "VGK", "WSH", "WPG"
            ];

            /**
             * fillDropdown - fills both home and away teams dropdown menus
             * with the team abbreviations from the 'teams' array.
             */
            function fillDropdown() {
                teams.forEach(team => {
                    homeSelect.add(new Option(team, team));
                    awaySelect.add(new Option(team, team));
                });
            }

            /**
             * fetchPrediction - async function that fetches the prediction data
             * sends an HTTP request to the express server (which proxies) and gets the request from the API.
             */
            async function fetchPrediction() {
                const home = homeSelect.value;
                const away = awaySelect.value;
                if (!home || !away) {
                    predictionResult.textContent = "Please select both teams.";
                    return;
                }

                try {
                    const response = await fetch(`/api/nhl/ml/predict?home=${home}&away=${away}`);
                    if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);

                    const data = await response.json();
                    const [homeWinProb, awayWinProb] = data.predictions[0];

                    predictionResult.innerHTML = `
                        <p><strong>${home} Win Probability:</strong> ${(homeWinProb * 100).toFixed(2)}%</p>
                        <p><strong>${away} Win Probability:</strong> ${(awayWinProb * 100).toFixed(2)}%</p>
                    `;
                } catch (error) {
                    predictionResult.textContent = "Error fetching prediction.";
                    console.error("Error fetching prediction:", error);
                }
            }

            getPredictionBtn.addEventListener("click", fetchPrediction);
            fillDropdown();
        });
    }
}
