const express = require('express'); 
const path = require('path'); 
const cors = require('cors'); 
const http = require('http'); 
const app = express(); 

const PORT = 3000; 

// enables CORS for allowing the frontend access data from the API.
app.use(cors());

// serves the html pages
app.use(express.static(path.join(__dirname, '..')));

// get endpoint for the api (proxy between frontend and the API)
app.get('/api/nhl/ml/predict', (req, res) => {
    // extract the home and away teams from the query parameters in the url.
    const { home, away } = req.query; 
    
    // error if a team is not provided
    if (!home || !away) {
        return res.status(400).json({ error: "Missing 'home' or 'away' team parameter" });
    }

    // construct the URL for the api, fills the team parameters.
    const predictionsAPI = `http://localhost:5000/api/nhl/ml/predict?home=${home}&away=${away}`;

    // sends a get request to the api
    http.get(predictionsAPI, (flaskRes) => {
        let data = '';

        // waits for the data from the api
        flaskRes.on('data', (chunk) => {
            data += chunk; // appends the data to a variable
        });

        // when the response is recieved, ...
        flaskRes.on('end', () => {
            try {
                // parse the received data (a string) into JSON.
                const jsonResponse = JSON.parse(data);
                // sends the json to the frontend.
                res.json(jsonResponse);
            } catch (error) {
                res.status(500).json({ error: "Invalid JSON from API" });
            }
        });
    }).on('error', (err) => {
        console.error("Error fetching predictions from API:", err.message);
        res.status(500).json({ error: "Error fetching predictions" });
    });
});

app.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
});
