const sendBtn = document.getElementById("sendBtn");
const userInput = document.getElementById("userInput");
const chatBox = document.getElementById("chatBox");

const API_URL = "http://127.0.0.1:5000/predict";

const emotionColors = {
  joy: "#FFF9C4",
  sadness: "#BBDEFB",
  anger: "#FFCDD2",
  fear: "#E1BEE7",
  love: "#F8BBD0",
  surprise: "#FFE0B2",
  neutral: "#E0E0E0",
  frustration: "#FFE082",
  hopeful: "#C8E6C9",
  content: "#DCEDC8"
};

sendBtn.addEventListener("click", sendMessage);
userInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter") sendMessage();
});

function addMessage(text, sender, emotion = null) {
  const bubble = document.createElement("div");
  bubble.classList.add("chat-bubble", sender);
  bubble.textContent = text;

  if (sender === "bot" && emotion) {
    const color = emotionColors[emotion.toLowerCase()] || "#F1F0F0";
    bubble.style.backgroundColor = color;

    const tag = document.createElement("div");
    tag.classList.add("emotion-tag");
    tag.textContent = `Emotion detected: ${emotion}`;
    bubble.appendChild(tag);
  }

  chatBox.appendChild(bubble);
  chatBox.scrollTop = chatBox.scrollHeight;
}

async function sendMessage() {
  const message = userInput.value.trim();
  if (!message) return;

  addMessage(message, "user");
  userInput.value = "";

  const typing = document.createElement("div");
  typing.classList.add("chat-bubble", "bot");
  typing.textContent = "🤖 Thinking...";
  chatBox.appendChild(typing);
  chatBox.scrollTop = chatBox.scrollHeight;

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });

    const data = await response.json();
    chatBox.removeChild(typing);

    if (data.predicted_emotion && data.bot_reply) {
      addMessage(data.bot_reply, "bot", data.predicted_emotion);
    } else if (data.error) {
      addMessage(`⚠️ Error: ${data.error}`, "bot");
    }
  } catch (err) {
    chatBox.removeChild(typing);
    addMessage("❌ Unable to connect to server. Make sure Flask is running.", "bot");
  }
}
