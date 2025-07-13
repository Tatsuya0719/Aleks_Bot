// src/main.tsx
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.tsx'; // This line imports your App component
import './index.css'; // This imports your main CSS, including Tailwind

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>
);