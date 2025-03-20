    // blocks the page unless the user is subscribed
    if (!sessionStorage.getItem("currentUser")) {
        window.location.href = "pricing.html";
    } else {
        let users = JSON.parse(sessionStorage.getItem("users")) || {};
        let currentUser = sessionStorage.getItem("currentUser");

        if (!users[currentUser]?.is_subscribed) {
            alert("You need a subscription to view this page.");
            window.location.href = "pricing.html";
        } else {
            document.addEventListener("DOMContentLoaded", () => {
                const homeSelect = document.getElementById("home-team");
                const awaySelect = document.getElementById("away-team");
                const ouInput = document.getElementById("ou-value"); 
                const predictionResult = document.getElementById("ml-result");  
                const ouResult = document.getElementById("ou-result");
                const getPredictionBtn = document.getElementById("get-prediction");
                const getOUPredictionBtn = document.getElementById("get-ou-prediction");

                // list of the nhl team abbreviations for the dropdown menu
                const teams = [
                    "ANA", "ARI", "BOS", "BUF", "CGY", "CAR", "CHI", "COL", "CBJ", "DAL",
                    "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NSH", "NJD", "NYI", "NYR",
                    "OTT", "PHI", "PIT", "SEA", "SJS", "STL", "TBL", "TOR", "VAN", "VGK", "WSH", "WPG"
                ];

                /**
                 * fillDropdown - fills the dropdown menus
                 * with the team abbreviations from the teams array.
                 */
                function fillDropdown() {
                    teams.forEach(team => {
                        homeSelect.add(new Option(team, team));
                        awaySelect.add(new Option(team, team));
                    });
                }

                /**
                 * fetchPrediction - async function that fetches the Moneyline prediction data
                 * sends an http request to the express server and retrieves the response from the API.
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
                        predictionResult.textContent = "Error fetching ML prediction.";
                        console.error("Error fetching ML prediction:", error);
                    }
                }

                /**
                 * fetchOuPrediction - async function that fetches the OU prediction data
                 * Ssends an HTTP request to the express server and retrieves the response from the API.
                 */
                async function fetchOUPrediction() {
                    const home = homeSelect.value;
                    const away = awaySelect.value;
                    const ouValue = ouInput.value;  

                    if (!home || !away || !ouValue) {
                        ouResult.textContent = "Please select both teams and enter an Over/Under value.";
                        return;
                    }

                    try {
                        const response = await fetch(`/api/nhl/ou/predict?home=${home}&away=${away}&ou=${ouValue}`);
                        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);

                        const data = await response.json();

                        // check if the response from the api is in a proper structure
                        if (!data.predictions || !Array.isArray(data.predictions) || data.predictions.length === 0 || data.predictions[0].length < 2) {
                            throw new Error("Invalid API response structure");
                        }

                        const [underProb, overProb] = data.predictions[0];  // get the over and under probabilities

                        // check if the probabilities are valid numbers
                        if (isNaN(underProb) || isNaN(overProb)) {
                            throw new Error("Received NaN from API");
                        }

                        ouResult.innerHTML = `
                            <p><strong>Over ${ouValue} Probability:</strong> ${(overProb * 100).toFixed(1)}%</p>
                            <p><strong>Under ${ouValue} Probability:</strong> ${(underProb * 100).toFixed(1)}%</p>
                        `;
                    } catch (error) {
                        ouResult.textContent = "Error fetching OU prediction.";
                        console.error("Error fetching OU prediction:", error);
                    }
                }

                // event listeners for the buttons
                getPredictionBtn.addEventListener("click", fetchPrediction);
                getOUPredictionBtn.addEventListener("click", fetchOUPrediction);

                // initalize the dropdown menu
                fillDropdown();
            });
        }
    }
