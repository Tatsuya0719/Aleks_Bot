import { useState, useRef, useEffect } from 'react';

// Assuming Tailwind CSS is available
// You might need to add a meta viewport tag in your public/index.html for responsiveness
// <meta name="viewport" content="width=device-width, initial-scale=1.0">

function App() {
  const [messages, setMessages] = useState<{ text: string; sender: 'user' | 'aleks'; isStreaming?: boolean; }[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Use the relative path, as Netlify will proxy this to your backend
  const API_BASE_URL = '/api'; 

  // Scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async () => {
    if (inputMessage.trim() === '') return;

    const userMessage = inputMessage;
    setMessages(prevMessages => [...prevMessages, { text: userMessage, sender: 'user' }]);
    setInputMessage('');
    setIsLoading(true);

    // Add a placeholder for Aleks's streaming response
    setMessages(prevMessages => [...prevMessages, { text: '', sender: 'aleks', isStreaming: true }]);

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: userMessage }),
      });

      // Handle non-streaming responses (e.g., document requests, errors)
      // Check for 'application/json' first as streaming response will be 'text/event-stream'
      if (response.headers.get('Content-Type')?.includes('application/json')) {
        const data = await response.json();
        setIsLoading(false);
        setMessages(prevMessages => 
          prevMessages.map(msg => 
            msg.isStreaming ? { ...msg, text: data.response || data.message, isStreaming: false } : msg
          )
        );
        // You might want to handle document_request type responses here specifically
        // For now, it will just display the message.
        return;
      }

      // Handle streaming responses
      if (!response.body) {
        throw new Error('Response body is null, cannot stream.');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let receivedText = '';

      // Update the streaming message in real-time
      setMessages(prevMessages => 
        prevMessages.map(msg => 
          msg.isStreaming ? { ...msg, text: '', isStreaming: true } : msg
        )
      );

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        const chunk = decoder.decode(value, { stream: true });
        receivedText += chunk;
        setMessages(prevMessages => 
          prevMessages.map(msg => 
            msg.isStreaming ? { ...msg, text: receivedText, isStreaming: true } : msg
          )
        );
      }

      // Mark streaming as complete
      setMessages(prevMessages => 
        prevMessages.map(msg => 
          msg.isStreaming ? { ...msg, isStreaming: false } : msg
        )
      );

      setIsLoading(false);

    } catch (error) {
      console.error('Failed to fetch Aleks API:', error);
      setIsLoading(false);
      setMessages(prevMessages => 
        prevMessages.map(msg => 
          // FIX: Removed an extra '}' here which caused the compilation error
          msg.isStreaming ? { ...msg, text: `Error: Failed to fetch. Please ensure the Aleks AI API server is running. (${error instanceof Error ? error.message : String(error)})`, isStreaming: false } : msg
        )
      );
    }
  };

  return (
    // FIX: Removed bg-gradient-to-br from App.tsx's root div.
    // The modal in index.html already provides the background and centering.
    // This div now only defines the content's size and flex behavior within the modal.
    <div className="bg-white rounded-xl shadow-2xl p-6 w-full max-w-2xl flex flex-col h-[80vh] md:h-[90vh]"> 
      {/* Header */}
      <div className="flex justify-between items-center pb-4 border-b border-gray-200">
        <h1 className="text-2xl font-bold text-gray-800">Aleks - AI Legal Assistant</h1>
        <button 
          onClick={() => setMessages([])} 
          className="text-gray-500 hover:text-gray-700 focus:outline-none"
          aria-label="Clear chat"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Message Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-gray-500 text-center">
            <p className="text-lg mb-2">Hi there! I am Aleks, your AI legal assistant for Filipino citizens. How can I help you today?</p>
            <button 
              onClick={() => handleSendMessage()} // This needs to be adjusted if it's meant to be a pre-defined message
              className="px-4 py-2 bg-yellow-400 text-white rounded-lg hover:bg-yellow-500 transition-colors shadow"
            >
              Try Aleks Now
            </button>
          </div>
        )}
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[70%] px-4 py-2 rounded-lg shadow-md ${
                msg.sender === 'user'
                  ? 'bg-orange-400 text-white'
                  : 'bg-gray-100 text-gray-800'
              }`}
            >
              {msg.text}
              {msg.isStreaming && (
                <span className="animate-pulse text-gray-500">_</span> // Simple blinking cursor
              )}
            </div>
          </div>
        ))}
        {isLoading && messages[messages.length - 1]?.sender !== 'aleks' && (
          <div className="flex justify-start">
            <div className="max-w-[70%] px-4 py-2 rounded-lg shadow-md bg-gray-100 text-gray-800">
              <div className="flex space-x-1">
                <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0s' }}></div>
                <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="pt-4 border-t border-gray-200 flex">
        <input
          type="text"
          className="flex-1 p-3 border border-gray-300 rounded-l-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
          placeholder="Type your message..."
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          onKeyPress={(e) => {
            if (e.key === 'Enter') {
              handleSendMessage();
            }
          }}
          disabled={isLoading}
        />
        <button
          onClick={handleSendMessage}
          className="px-6 py-3 bg-yellow-400 text-white rounded-r-lg hover:bg-yellow-500 transition-colors shadow-md focus:outline-none focus:ring-2 focus:ring-yellow-400"
          disabled={isLoading}
        >
          Send
        </button>
      </div>
    </div>
  );
}

export default App;
