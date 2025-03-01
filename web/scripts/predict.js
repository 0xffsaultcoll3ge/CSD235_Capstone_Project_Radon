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
     * fillDropdown fills both home and away teams
     * dropdown menus with the team abbreviations from the 'teams' array.
     * iterates over the array with a forEach loop
     * each team abbreviation creates a new option element using the constructor
     * and adds it to both dropdown menus.
     */
    function fillDropdown() {
        teams.forEach(team => {
            const option1 = new Option(team, team);
            const option2 = new Option(team, team); 
            homeSelect.add(option1);
            awaySelect.add(option2);
        });
    }

    /**
     * fetchPrediction - async function that fetches the prediction data 
     * sends an http request to the express server (which proxies) and gets the request from the API
     *  the function waits for the functions promise fetch and json response promise to be fufilled.
     */
    async function fetchPrediction() {
        const home = homeSelect.value;
        const away = awaySelect.value;
        if (!home || !away) {
            predictionResult.textContent = "Please select both teams.";
            return;
        }

        try {
            // fetch API to send a get request to the express server
            const response = await fetch(`/api/nhl/ml/predict?home=${home}&away=${away}`);
            
            // checks the response and throws and error if not.
            if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);

            // parses the json
            const data = await response.json();

            // extracts the values from the json response to be used.
            const [homeWinProb, awayWinProb] = data.predictions[0];

            // updates the html to display the result of the prediction (converts into a percentage)
            predictionResult.innerHTML = `
                <p><strong>${home} Win Probability:</strong> ${(homeWinProb * 100).toFixed(2)}%</p>
                <p><strong>${away} Win Probability:</strong> ${(awayWinProb * 100).toFixed(2)}%</p>
            `;
        } catch (error) {
            // checks the response and throws and error if not..
            predictionResult.textContent = "Error fetching prediction.";
            console.error("Error fetching prediction:", error);
        }
    }
    // event listener for button click
    getPredictionBtn.addEventListener("click", fetchPrediction);

    // calls the fillDropdown function to fill the dropdown menus with teams
    fillDropdown();
});
