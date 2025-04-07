document.addEventListener("DOMContentLoaded", async () => {
    const AUTH_API = "http://localhost:5000/api/user";

    const teamSelect = document.getElementById("team-select");
    const historyResult = document.getElementById("history-result");
    const getHistoryBtn = document.getElementById("get-history");

    if (!teamSelect) {
        console.error("Team dropdown not found!");
        return;
    }

    /**
     * fills the team dropdown with all team options
     */
    function fillTeamDropdown() {
        const teams = [
            "ANA", "ARI", "BOS", "BUF", "CGY", "CAR", "CHI", "COL", "CBJ", "DAL",
            "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NSH", "NJD", "NYI", "NYR",
            "OTT", "PHI", "PIT", "SEA", "SJS", "STL", "TBL", "TOR", "VAN", "VGK", "WSH", "WPG"
        ];

        teamSelect.innerHTML = '<option value="">Select Team</option>';
        teams.forEach(team => {
            teamSelect.add(new Option(team, team));
        });
    }

    /**
     * fetches the historical data for the selected team
     */
    async function fetchTeamHistory() {
        const team = teamSelect.value;

        if (!team) {
            historyResult.textContent = "Please select a team.";
            return;
        }

        try {
            const response = await fetch(`http://localhost:5000/api/nhl/teams/data?team=${team}`);
            const data = await response.json();
            
            // displaying team history data in the scrollable box
            historyResult.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
        } catch (error) {
            historyResult.textContent = "Error fetching team history.";
            console.error(error);
        }
    }

    /**
     * checks if the user is subscribed
     * fetches user data from the backend api
     * if the user is not subscribed, redirects them to the pricing page
     * saves user data in session storage for future use
     */
    async function checkSubscription() {
        try {
            const response = await fetch(AUTH_API, { credentials: "include" }); // sends request with credentials
            if (!response.ok) throw new Error("failed to fetch user data.");

            const user = await response.json(); // parses the json data

            if (!user.is_subscribed) {
                window.location.href = "pricing.html"; // redirects to pricing page if not subscribed
                return;
            }

            sessionStorage.setItem("currentUser", JSON.stringify(user)); // stores user data in session storage
        } catch (error) {
            console.error("error checking subscription:", error);
            window.location.href = "pricing.html"; // redirects if error occurs
        }
    }

    getHistoryBtn.addEventListener("click", fetchTeamHistory);

    // check subscription status first, then fill the team dropdown and enable history functionality
    await checkSubscription();
    fillTeamDropdown();
});
