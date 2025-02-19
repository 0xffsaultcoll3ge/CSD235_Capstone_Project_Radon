const express = require('express'); 
const path = require('path'); 
const cors = require('cors'); 

const app = express(); 
const PORT = 3000; 

// enables cors for frontend requests (this allows the browser to get data from the api)
app.use(cors());

// serve the frontend to the server
app.use(express.static(__dirname));

// mock nhl api predictions
app.get('/api/nhl/ml/predict', (req, res) => {
    console.log("API request received at /api/nhl/ml/predict"); 

    // sends mock prediction data as json response
    res.json([
        { home: "TOR", away: "MTL", prediction: "TOR 60%" },
        { home: "NYR", away: "BOS", prediction: "BOS 55%" },
        { home: "CHI", away: "LAK", prediction: "LAK 58%" }
    ]);
});

app.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`); 
});
