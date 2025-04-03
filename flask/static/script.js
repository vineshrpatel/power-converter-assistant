document.addEventListener('keydown', function(event) {
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        event.preventDefault();
        document.getElementById('send-button').click();
    }
});

async function sendMessage() {
    let inputField = document.getElementById("user-input");
    let chatBox = document.getElementById("chat-box");

    let userMessage = inputField.value.trim();
    if (userMessage === "") return;

    chatBox.innerHTML += `<div class="user-message"><strong>You:</strong> ${userMessage}</div>`;

    inputField.disabled = true;
    inputField.value = "";

    let tempMessage = document.createElement("div");
    tempMessage.classList.add("bot-message", "temp-message");
    tempMessage.innerHTML = `<strong>Power Converter Assistant:</strong> <span style="color: #808080;">Generating response...</span>`;
    chatBox.appendChild(tempMessage);
    chatBox.scrollTop = chatBox.scrollHeight;

    try {
        let response = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: userMessage }),
        });

        let data = await response.json();
        let botMessage = data.response || "Error: Unable to get response.";

        tempMessage.remove();
        chatBox.innerHTML += `<div class="bot-message"><strong>Power Converter Assistant:</strong> ${botMessage}</div>`;
    } catch (error) {
        tempMessage.innerHTML = `<strong>Power Converter Assistant:</strong> Error fetching response. Please try again.`;
    } finally {
        inputField.disabled = false;
        inputField.focus();
    }

    chatBox.scrollTop = chatBox.scrollHeight;
}

async function runGUI() {
    let chatBox = document.getElementById("chat-box");

    // Ask for converter type every time
    selectedConverter = prompt("Enter DC-DC power converter:");
    if (!selectedConverter) {
        chatBox.innerHTML += `<div class="system-message"><strong>System:</strong> No converter selected. GUI launch canceled.</div>`;
        return;
    }
    
    selectedConverter = selectedConverter.trim().toLowerCase();
    chatBox.innerHTML += `<div class="system-message"><span style="color: #808080;"><strong>System:</strong> Selected Converter: ${selectedConverter.toUpperCase()}</span></div>`;
    chatBox.innerHTML += `<div class="system-message"><span style="color: #808080;"><strong>System:</strong> Launching ${selectedConverter.toUpperCase()} Converter GUI...</span></div>`;
    chatBox.scrollTop = chatBox.scrollHeight;

    try {
        let response = await fetch("/run_gui", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ converter: selectedConverter }),
        });

        let data = await response.json();
        let guiMessage = data.response || `Error: ${data.error}`;

        chatBox.innerHTML += `<div class="system-message"><strong>System:</strong> ${guiMessage}</div>`;
    } catch (error) {
        chatBox.innerHTML += `<div class="system-message"><strong>System:</strong> Failed to launch ${selectedConverter} GUI.</div>`;
    }

    chatBox.scrollTop = chatBox.scrollHeight;
}
