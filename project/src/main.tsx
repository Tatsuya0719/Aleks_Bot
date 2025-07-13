import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.tsx'; // This is your Chatbot App component
import './index.css'; // Your main CSS for the React app

// IMPORTANT: We are now mounting the React App into the 'ai-chatbot-root' div
// which is located inside the chatbot modal in index.html.
const chatbotRootElement = document.getElementById('ai-chatbot-root');

if (chatbotRootElement) {
  createRoot(chatbotRootElement).render(
    <StrictMode>
      <App />
    </StrictMode>
  );
} else {
  console.error("Could not find element with id 'ai-chatbot-root' to mount React chatbot.");
}