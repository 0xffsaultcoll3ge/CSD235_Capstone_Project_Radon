document.addEventListener("DOMContentLoaded", () => {
    fetchPredictions(); // calls the fetchprediction function
});

/**
 * fetchPredictions - gets the NHL event predictions from the API and updates the ui.
 * uses the fetch api, which returns a promise.
 */
function fetchPredictions() {
    fetch('/api/nhl/ml/predict') // fetch data from the mock API endpoint (returns a promise)
        .then(response => { 
            if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
            return response.json(); // converts the response to json (returns a **promise**)
        })
        .then(data => updatePredictions(data)) // promise - waits for the json conversion then updates ui
        .catch(error => console.error('Error fetching predictions:', error)); 
}

/**
 * updatePredictions - dynamically updates the predictions page with data
 * @param {Array} predictions - array of prediction objects from the api
 */
function updatePredictions(predictions) {
    const predictionsContainer = document.querySelector(".predictions-container");
    predictionsContainer.innerHTML = ""; 

    // loops through each prediction and creates a ui element
    predictions.forEach(game => {
        const box = document.createElement("div"); 
        box.classList.add("prediction-box"); 

        // sets the html with prediction data
        box.innerHTML = `
            <h2>${game.home} vs ${game.away}</h2>
            <p><strong>Prediction:</strong> ${game.prediction}</p>
        `;

        predictionsContainer.appendChild(box); // append the new box with the data to the container
    });
}
