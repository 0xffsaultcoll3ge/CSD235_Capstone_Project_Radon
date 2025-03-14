document.addEventListener("DOMContentLoaded", () => {
    // account icon that opens the dropdown menu (for raghav so he knows what each element is)
    const accountIcon = document.getElementById("accountIcon");
    // dropdown menu containing login, register, and account details (for raghav so he knows what each element is)
    const accountDropdown = document.getElementById("accountDropdown");
    // input field for login email (for raghav so he knows what each element is)
    const loginEmail = document.getElementById("loginEmail");
    // input field for login password (for raghav so he knows what each element is)
    const loginPassword = document.getElementById("loginPassword");
    // input field for registration email (for raghav so he knows what each element is)
    const registerEmail = document.getElementById("registerEmail");
    // input field for registration password (for raghav so he knows what each element is)
    const registerPassword = document.getElementById("registerPassword");
    // displays the currently logged-in user's email or "Account" (for raghav so he knows what each element is)
    const userDisplay = document.getElementById("userDisplay");
    // container for login form elements (for raghav so he knows what each element is)
    const authForms = document.getElementById("authForms");
    // container for registration form elements (for raghav so he knows what each element is)
    const registerForm = document.getElementById("registerForm");
    // logout button, only visible when a user is logged in (for raghav so he knows what each element is)
    const logoutButton = document.getElementById("logoutButton");
    // displays the subscription status of the logged-in user (for raghav so he knows what each element is)
    const subscriptionStatus = document.getElementById("subscriptionStatus");

    // persistent test account (always exists)
    let testAccount = { password: "test", subscribed: true };
    let savedTestAccount = JSON.parse(localStorage.getItem("testAccount"));

    // ensures the test account always exists in localStorage
    if (!savedTestAccount || savedTestAccount.password !== "test") {
        localStorage.setItem("testAccount", JSON.stringify(testAccount));
    }

    // users only exist in sessionStorage (cleared on restart)
    let users = JSON.parse(sessionStorage.getItem("users")) || {};

    // ensures the test account is always available in sessionStorage
    users["test@test.com"] = JSON.parse(localStorage.getItem("testAccount"));
    sessionStorage.setItem("users", JSON.stringify(users));

    // stores the currently logged-in user, or null if no user is logged in
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
            userDisplay.innerText = `Logged in as: ${currentUser}`;
            authForms.style.display = "none";
            registerForm.style.display = "none";
            subscriptionStatus.innerText = `Subscription: ${users[currentUser].subscribed ? "Yes" : "No"}`;
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

        if (users[email] && users[email].password === password) {
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
        const email = registerEmail.value.trim();
        const password = registerPassword.value.trim();

        if (!email || !password) {
            alert("Please fill in all fields.");
            return;
        }

        if (users[email]) {
            alert("An account with this email already exists.");
            return;
        }

        users[email] = { password, subscribed: false };
        sessionStorage.setItem("users", JSON.stringify(users));
        alert("Registration successful. Please log in.");
        registerForm.style.display = "none";
        authForms.style.display = "block";
    };

    /**
     * Handles user logout
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
