// Event Listeners for sending messages
document.getElementById("send-btn").addEventListener("click", sendMessage);
document.getElementById("user-input").addEventListener("keypress", function (event) {
    if (event.key === "Enter") sendMessage(); // Trigger sendMessage on Enter key
});

function sendMessage() {
    const userInput = document.getElementById("user-input");
    const message = userInput.value.trim();

    if (message) {
        appendMessage("user", message); // Append user message to chat

        // Fetch response and suggestions from the backend
        fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ message: message }),
        })
            .then((response) => {
                if (!response.ok) {
                    throw new Error("Network response was not ok");
                }
                return response.json();
            })
            .then((data) => {
                // Append chatbot response to chat
                appendMessage("bot", data.response || "Sorry, I couldn't understand that.");

                // Update suggestions dynamically
                updateSuggestions(data.related || []);
            })
            .catch((error) => {
                console.error("Fetch error:", error);
                appendMessage("bot", "Oops! Something went wrong. Please try again later.");
            });

        userInput.value = ""; // Clear the input field
    }
}

function appendMessage(sender, message) {
    const chatBox = document.getElementById("chat-box");
    const msgDiv = document.createElement("div");
    msgDiv.classList.add("message", sender);
    msgDiv.textContent = message;
    chatBox.appendChild(msgDiv);

    // Auto-scroll to the latest message
    chatBox.scrollTop = chatBox.scrollHeight;
}

function updateSuggestions(related) {
    const suggestionsDiv = document.getElementById("suggestions");
    suggestionsDiv.innerHTML = ""; // Clear previous suggestions

    if (related && related.length > 0) {
        related.forEach((item) => {
            const suggestionButton = document.createElement("button");
            suggestionButton.classList.add("suggestion-btn");
            suggestionButton.textContent = item.question; // Display related question
            suggestionButton.addEventListener("click", () => {
                document.getElementById("user-input").value = item.question; // Pre-fill input
                sendMessage(); // Send the suggested query
            });
            suggestionsDiv.appendChild(suggestionButton);
        });
    } else {
        // Add a placeholder if no suggestions are available
        const noSuggestion = document.createElement("div");
        noSuggestion.textContent = "No suggestions available.";
        noSuggestion.classList.add("no-suggestions");
        suggestionsDiv.appendChild(noSuggestion);
    }
}
