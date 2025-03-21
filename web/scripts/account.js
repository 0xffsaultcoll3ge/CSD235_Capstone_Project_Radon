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

    // test account 
    let testAccount = {
        first_name: "Test",
        last_name: "User",
        email: "test@test.com",
        password_hash: "test",
        is_subscribed: true
    };

    let savedTestAccount = JSON.parse(localStorage.getItem("testAccount"));

    // makes sure the test account exists 
    if (!savedTestAccount || savedTestAccount.password_hash !== "test") {
        localStorage.setItem("testAccount", JSON.stringify(testAccount));
    }

    // users only exist in session storage
    let users = JSON.parse(sessionStorage.getItem("users")) || {};
    users["test@test.com"] = JSON.parse(localStorage.getItem("testAccount")); 
    sessionStorage.setItem("users", JSON.stringify(users));

    // stores the currently logged in user
    let currentUser = sessionStorage.getItem("currentUser") || null;

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
        users = JSON.parse(sessionStorage.getItem("users"));

        if (currentUser && users[currentUser]) {
            userDisplay.innerText = `Logged in as: ${users[currentUser].first_name} ${users[currentUser].last_name}`;
            authForms.style.display = "none";
            registerForm.style.display = "none";
            subscriptionStatus.innerText = `Subscription: ${users[currentUser].is_subscribed ? "Yes" : "No"}`;
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

    updateAccountUI(); // ensures UI is updated on page load

    /**
     * Handles user login
     */
    window.handleLogin = function () {
        users = JSON.parse(sessionStorage.getItem("users"));
        const email = loginEmail.value.trim();
        const password = loginPassword.value.trim();

        if (!email || !password) {
            alert("Please fill in all fields.");
            return;
        }

        if (users[email] && users[email].password_hash === password) {  
            currentUser = email;
            sessionStorage.setItem("currentUser", email);
            updateAccountUI();
        } else {
            alert("Incorrect email or password.");
        }
    };

    /**
     * Handles user registration
     */
    window.handleRegister = function () {
        users = JSON.parse(sessionStorage.getItem("users"));
        const firstName = registerFirstName.value.trim();  
        const lastName = registerLastName.value.trim();    
        const email = registerEmail.value.trim();
        const password = registerPassword.value.trim();

        if (!firstName || !lastName || !email || !password) { 
            alert("Please fill in all fields.");
            return;
        }

        if (users[email]) {
            alert("An account with this email already exists.");
            return;
        }

        // stores the user with proper database fields
        users[email] = {
            first_name: firstName,
            last_name: lastName,
            email: email,
            password_hash: password,  
            is_subscribed: false  // default is false
        };

        sessionStorage.setItem("users", JSON.stringify(users));
        alert("Registration successful. Please log in.");
        registerForm.style.display = "none";
        authForms.style.display = "block";
    };

    /**
     * handles user logout
     */
    window.handleLogout = function () {
        currentUser = null;
        sessionStorage.removeItem("currentUser");
        updateAccountUI();
    };

    // switches from login to register form
    document.getElementById("showRegister").addEventListener("click", () => {
        authForms.style.display = "none";
        registerForm.style.display = "block";
    });

    // switches from register to login form
    document.getElementById("showLogin").addEventListener("click", () => {
        registerForm.style.display = "none";
        authForms.style.display = "block";
    });

    // attaches event listeners to login, register, and logout buttons
    document.getElementById("loginButton").addEventListener("click", handleLogin);
    document.getElementById("registerButton").addEventListener("click", handleRegister);
    document.getElementById("logoutButton").addEventListener("click", handleLogout);
});
