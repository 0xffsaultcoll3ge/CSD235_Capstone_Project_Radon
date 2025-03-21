document.addEventListener("DOMContentLoaded", () => {
    const accountIcon = document.getElementById("accountIcon");
    const accountDropdown = document.getElementById("accountDropdown");
    const loginEmail = document.getElementById("loginEmail");
    const loginPassword = document.getElementById("loginPassword");
    const registerFirstName = document.getElementById("registerFirstName"); 
    const registerLastName = document.getElementById("registerLastName");    
    const registerEmail = document.getElementById("registerEmail");
    const registerPassword = document.getElementById("registerPassword");
    const userDisplay = document.getElementById("userDisplay");
    const authForms = document.getElementById("authForms");
    const registerForm = document.getElementById("registerForm");
    const logoutButton = document.getElementById("logoutButton");
    const subscriptionStatus = document.getElementById("subscriptionStatus");

    // Stores the currently logged-in user
    let currentUser = JSON.parse(sessionStorage.getItem("currentUser")) || null;

    /**
     * Toggles the visibility of the account dropdown menu
     */
    accountIcon.addEventListener("click", (event) => {
        event.stopPropagation();
        accountDropdown.classList.toggle("show");
        updateAccountUI();
    });

    /**
     * Closes the dropdown when clicking outside of it
     */
    document.addEventListener("click", (event) => {
        if (!accountDropdown.contains(event.target) && event.target !== accountIcon) {
            accountDropdown.classList.remove("show");
        }
    });

    /**
     * Updates the UI based on whether a user is logged in or not
     */
    function updateAccountUI() {
        if (currentUser) {
            userDisplay.innerText = `Logged in as: ${currentUser.first_name} ${currentUser.last_name}`;
            authForms.style.display = "none";
            registerForm.style.display = "none";
            subscriptionStatus.innerText = `Subscription: ${currentUser.is_subscribed ? "Yes" : "No"}`;
            subscriptionStatus.style.display = "block";
            logoutButton.style.display = "block";
        } else {
            userDisplay.innerText = "Account";
            authForms.style.display = "block";
            registerForm.style.display = "none";
            subscriptionStatus.style.display = "none";
            logoutButton.style.display = "none";
        }
    }

    updateAccountUI(); // Ensures UI is updated on page load

    /**
     * Handles user login
     */
    window.handleLogin = async function () {
        const email = loginEmail.value.trim();
        const password = loginPassword.value.trim();

        if (!email || !password) {
            alert("Please fill in all fields.");
            return;
        }

        try {
            const response = await fetch('/api/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    email: email,
                    password: password
                }),
            });

            const data = await response.json();
            if (response.ok) {
                currentUser = data.user;
                sessionStorage.setItem("currentUser", JSON.stringify(currentUser));
                updateAccountUI();
                alert(data.message);
            } else {
                alert(data.error);
            }
        } catch (error) {
            alert("An error occurred. Please try again.");
            console.error(error);
        }
    };

    /**
     * Handles user registration
     */
    window.handleRegister = async function () {
        const firstName = registerFirstName.value.trim();  
        const lastName = registerLastName.value.trim();    
        const email = registerEmail.value.trim();
        const password = registerPassword.value.trim();

        if (!firstName || !lastName || !email || !password) { 
            alert("Please fill in all fields.");
            return;
        }

        try {
            const response = await fetch('/api/register', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    first_name: firstName,
                    last_name: lastName,
                    email: email,
                    password: password
                }),
            });

            const data = await response.json();
            if (response.ok) {
                alert(data.message);
                registerForm.style.display = "none";
                authForms.style.display = "block";
            } else {
                alert(data.error);
            }
        } catch (error) {
            alert("An error occurred. Please try again.");
            console.error(error);
        }
    };

    /**
     * Handles user logout
     */
    window.handleLogout = async function () {
        try {
            const response = await fetch('/api/logout', {
                method: 'POST',
            });

            const data = await response.json();
            if (response.ok) {
                currentUser = null;
                sessionStorage.removeItem("currentUser");
                updateAccountUI();
                alert(data.message);
            } else {
                alert(data.error);
            }
        } catch (error) {
            alert("An error occurred. Please try again.");
            console.error(error);
        }
    };

    // Switches from login to register form
    document.getElementById("showRegister").addEventListener("click", () => {
        authForms.style.display = "none";
        registerForm.style.display = "block";
    });

    // Switches from register to login form
    document.getElementById("showLogin").addEventListener("click", () => {
        registerForm.style.display = "none";
        authForms.style.display = "block";
    });

    // Attaches event listeners to login, register, and logout buttons
    document.getElementById("loginButton").addEventListener("click", handleLogin);
    document.getElementById("registerButton").addEventListener("click", handleRegister);
    document.getElementById("logoutButton").addEventListener("click", handleLogout);
});