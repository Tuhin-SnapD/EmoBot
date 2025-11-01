const sendBtn = document.getElementById("sendBtn");
const userInput = document.getElementById("userInput");
const chatBox = document.getElementById("chatBox");

const API_URL = "http://127.0.0.1:5000/predict";

// 🎨 Emotion color map
const emotionColors = {
  joy: "#FFF9C4",
  sadness: "#BBDEFB",
  anger: "#FFCDD2",
  fear: "#E1BEE7",
  love: "#F8BBD0",
  surprise: "#FFE0B2",
  neutral: "#E0E0E0",
};

// 🧠 Store predicted emotions per message
let emotionHistory = [];
const MESSAGE_LIMIT = 5; // after how many user messages to summarize mood

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
    tag.textContent = `Detected emotion: ${emotion}`;
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

    if (data.predicted_emotion) {
      const emotion = data.predicted_emotion;
      emotionHistory.push(emotion);
      addMessage(
        `I sense you're feeling ${emotion}.`,
        "bot",
        emotion
      );

      // 💡 After 5 messages, show overall mood summary
      if (emotionHistory.length === MESSAGE_LIMIT) {
        showMoodSummary();
        emotionHistory = []; // reset after summary
      }
    } else if (data.error) {
      addMessage(`⚠️ Error: ${data.error}`, "bot");
    }
  } catch (err) {
    chatBox.removeChild(typing);
    addMessage("❌ Unable to connect to server. Make sure Flask is running.", "bot");
  }
}

// 🧩 Show overall mood summary
function showMoodSummary() {
  if (emotionHistory.length === 0) return;

  const freq = {};
  for (const emo of emotionHistory) {
    freq[emo] = (freq[emo] || 0) + 1;
  }

  const overallMood = Object.keys(freq).reduce((a, b) =>
    freq[a] > freq[b] ? a : b
  );

  // Suggestion tips based on mood
  const tips = {
    joy: "Keep smiling and spread positivity! 🌻",
    sadness: "Take a short walk or listen to your favorite music. 🎧",
    anger: "Take a deep breath — maybe a short break will help. 🌿",
    fear: "You’re stronger than your fears. Believe in yourself 💪",
    love: "Aww, stay kind and caring! 💖",
    surprise: "Wow! Sounds exciting — tell me more. 😲",
    neutral: "Keep going! You're doing fine. 💫",
  };

  const summaryColor = emotionColors[overallMood] || "#f1f0f0";

  const summary = document.createElement("div");
  summary.classList.add("chat-bubble", "bot");
  summary.style.backgroundColor = summaryColor;
  summary.innerHTML = `
    <strong>💬 Overall Mood: ${overallMood.toUpperCase()}</strong><br>
    <em>${tips[overallMood]}</em>
  `;
  chatBox.appendChild(summary);
  chatBox.scrollTop = chatBox.scrollHeight;
}
