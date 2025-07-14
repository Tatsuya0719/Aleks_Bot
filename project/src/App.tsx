import React, { useState, useEffect, useRef, FormEvent } from 'react';

// Define the API base URL for your FastAPI server
// const API_BASE_URL = 'http://localhost:8000';
const API_BASE_URL = 'http://10.147.18.65:8000';

// Define types for messages and placeholders for better type safety (optional but good practice)
interface Message {
  sender: 'user' | 'aleks';
  text: string;
  type: 'text' | 'document_request' | 'document_generated';
  additionalData?: {
    sources?: Array<{ source: string; startIndex: string; snippet: string }>;
    documentType?: string;
    placeholdersToFill?: Array<{ name: string; description: string }>;
    generatedDocumentPreview?: string;
  };
}

interface Placeholder {
  name: string;
  description: string;
}

function App() {
  // We no longer manage modal visibility here, as index.html handles it
  // This component now *is* the content inside the modal.
  const [isDocumentFillModalOpen, setIsDocumentFillModalOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      sender: 'aleks',
      text: 'Hi there! I am Aleks, your AI legal assistant for Filipino citizens. How can I help you today?',
      type: 'text',
    },
  ]);
  const [userInput, setUserInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentDocumentType, setCurrentDocumentType] = useState('');
  const [currentPlaceholders, setCurrentPlaceholders] = useState<Placeholder[]>([]);

  const chatMessagesRef = useRef<HTMLDivElement>(null);
  const userInputRef = useRef<HTMLTextAreaElement>(null);
  const documentFillFormRef = useRef<HTMLFormElement>(null);


  // Scroll to bottom of chat messages
  useEffect(() => {
    if (chatMessagesRef.current) {
      chatMessagesRef.current.scrollTop = chatMessagesRef.current.scrollHeight;
    }
  }, [messages]); // Scroll on new messages

  // Focus input when the component mounts (i.e., modal opens)
  useEffect(() => {
    if (userInputRef.current) {
      userInputRef.current.focus();
    }
  }, []); // Empty dependency array means this runs once on mount


  const addMessage = (message: Message) => {
    setMessages(prevMessages => [...prevMessages, message]);
  };

  const sendMessageToAleks = async () => {
    const question = userInput.trim();
    if (!question || isLoading || isDocumentFillModalOpen) return;

    addMessage({ sender: 'user', text: question, type: 'text' });
    setUserInput(''); // Clear input
    if (userInputRef.current) {
      userInputRef.current.style.height = 'auto'; // Reset textarea height
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: question }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'An unknown error occurred.');
      }

      const data = await response.json(); // Data from FastAPI

      if (data.type === 'document_request') {
        // Handle document request from API
        addMessage({
          sender: 'aleks',
          text: data.message,
          type: 'document_request',
          additionalData: {
            documentType: data.document_type,
            placeholdersToFill: data.placeholders_to_fill
          }
        });
        setCurrentDocumentType(data.document_type || '');
        setCurrentPlaceholders(data.placeholders_to_fill || []);
        setIsDocumentFillModalOpen(true);
      } else if (data.type === 'rag_response') {
        // Handle RAG response from API
        addMessage({
          sender: 'aleks',
          text: data.response,
          type: 'text',
          additionalData: { sources: data.sources },
        });
      } else if (data.type === 'text') {
        // Handle simple text response from API
        addMessage({ sender: 'aleks', text: data.response, type: 'text' });
      } else {
        // Fallback for unexpected response types
        addMessage({ sender: 'aleks', text: 'Received an unexpected response from the server.', type: 'text' });
      }

    } catch (err) {
      console.error('Error sending message to Aleks:', err);
      setError(`${(err as Error).message}. Please ensure the Aleks AI API server is running.`);
      addMessage({
        sender: 'aleks',
        text: `Error: ${(err as Error).message}. Please ensure the Aleks AI API server is running.`,
        type: 'text',
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleUserInputKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessageToAleks();
    }
  };

  const handleTextareaInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setUserInput(e.target.value);
    // Auto-resize textarea
    e.target.style.height = 'auto';
    e.target.style.height = `${e.target.scrollHeight}px`;
  };

  const handleDocumentFillSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    const form = e.currentTarget;
    const formData: { [key: string]: string } = {};
    currentPlaceholders.forEach(p => {
      const inputElement = form.elements.namedItem(p.name) as HTMLInputElement;
      if (inputElement) {
        formData[p.name] = inputElement.value;
      }
    });

    try {
      const response = await fetch(`${API_BASE_URL}/api/generate_document`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          template_key: currentDocumentType,
          filled_data: formData,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to generate document.');
      }

      const data = await response.json();
      addMessage({
        sender: 'aleks',
        text: data.message,
        type: 'document_generated',
        additionalData: { generatedDocumentPreview: data.generated_document_preview },
      });
      setIsDocumentFillModalOpen(false); // Close modal on success
      if (documentFillFormRef.current) {
        documentFillFormRef.current.reset(); // Reset form
      }
      setCurrentDocumentType('');
      setCurrentPlaceholders([]);
    } catch (err) {
      console.error('Error generating document:', err);
      setError(`Document Generation Error: ${(err as Error).message}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
      {/* This component IS the chat modal content.
          The modal itself and its open/close state is handled by index.html's JS. */}

      {/* Chat Messages Container */}
      <div id="chatMessages" ref={chatMessagesRef} className="flex-1 overflow-y-auto p-4 space-y-4 chat-messages">
        {messages.map((msg, index) => (
          <div key={index} className={`flex fade-in ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div
              className={`max-w-xs md:max-w-md px-4 py-2 rounded-lg shadow break-words ${
                msg.sender === 'user'
                  ? 'bg-orange-500 text-white' // User message color
                  : 'bg-lexibot-yellow-light text-gray-800 border border-gray-200' // Aleks message color
              }`}
            >
              <p className="font-semibold mb-1">{msg.sender === 'user' ? 'You' : 'Aleks'}</p>
              <div className="whitespace-pre-wrap" dangerouslySetInnerHTML={{ __html: msg.text }}></div>

              {msg.type === 'text' && msg.additionalData?.sources && msg.additionalData.sources.length > 0 && (
                <div className="mt-2 text-xs text-gray-600 border-t pt-2">
                  <p className="font-medium">Sources:</p>
                  <ul className="list-disc pl-4">
                    {msg.additionalData.sources.map((source, sIdx) => (
                      <li key={sIdx}>
                        <strong>{source.source}</strong> (Start Index: {source.startIndex || 'N/A'})<br />
                        <span className="italic text-gray-500 line-clamp-2">"{source.snippet}"</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {msg.type === 'document_generated' && msg.additionalData?.generatedDocumentPreview && (
                <div className="mt-2 text-sm bg-gray-50 p-3 rounded-md border border-gray-200 overflow-x-auto">
                  <p className="font-semibold mb-1">Generated Document Preview:</p>
                  <pre className="whitespace-pre-wrap text-gray-700 text-xs font-mono">
                    {msg.additionalData.generatedDocumentPreview}
                  </pre>
                  <p className="text-xs text-gray-500 mt-1">
                    (Document saved in your Python server's 'document_templates' folder)
                  </p>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Chat Input Area */}
      <div className="p-4 border-t flex items-center">
        <textarea
          id="userInput"
          ref={userInputRef}
          className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-lexibot-yellow resize-none h-auto overflow-hidden"
          placeholder="Type your message..."
          rows={1}
          value={userInput}
          onChange={handleTextareaInput}
          onKeyPress={handleUserInputKeyPress}
          disabled={isLoading || isDocumentFillModalOpen}
        ></textarea>
        <button
          id="sendButton"
          onClick={sendMessageToAleks}
          className="ml-4 px-6 py-2 bg-lexibot-yellow text-white rounded-lg hover:bg-lexibot-yellow-dark focus:outline-none focus:ring-2 focus:ring-lexibot-yellow disabled:opacity-50"
          disabled={!userInput.trim() || isLoading || isDocumentFillModalOpen}
        >
          Send
        </button>
      </div>
      {/* Loading and Error Indicators */}
      {isLoading && (
        <div id="loadingIndicator" className="text-center text-lexibot-yellow text-sm mb-2">
          <i className="fas fa-spinner fa-spin mr-2"></i> Thinking...
        </div>
      )}
      {error && (
        <div id="errorDisplay" className="bg-red-100 border border-red-400 text-red-700 px-3 py-2 rounded-md mx-4 mb-2 text-sm">
          Error: <span id="errorMessage">{error}</span>
        </div>
      )}

      {/* Document Fill Modal */}
      {isDocumentFillModalOpen && (
        <div
          id="documentFillModal"
          className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50 p-4"
        >
          <div className="bg-white rounded-lg shadow-xl w-full max-w-md max-h-[90vh] overflow-y-auto">
            <div className="flex justify-between items-center p-4 border-b">
              <h2 id="modalTitle" className="text-xl font-semibold text-gray-800">Fill Out {currentDocumentType} Template</h2>
              <button
                id="closeDocumentModalButton"
                onClick={() => { setIsDocumentFillModalOpen(false); setError(null); }}
                className="text-gray-500 hover:text-gray-700 text-2xl font-bold"
              >
                &times;
              </button>
            </div>
            <form id="documentFillForm" ref={documentFillFormRef} onSubmit={handleDocumentFillSubmit} className="p-4 space-y-4">
              <p className="text-sm text-gray-600">Please provide the following details:</p>
              <div id="placeholdersContainer" className="space-y-3">
                {currentPlaceholders.length === 0 ? (
                  <p className="text-sm text-gray-500">No specific placeholders found for this document type. Proceeding with general generation.</p>
                ) : (
                  currentPlaceholders.map((p, pIdx) => (
                    <div key={pIdx}>
                      <label htmlFor={p.name} className="block text-sm font-medium text-gray-700">
                        {p.description} <span className="text-red-500">*</span>
                      </label>
                      <input
                        type="text"
                        id={p.name}
                        name={p.name}
                        required
                        className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-lexibot-yellow-dark focus:border-lexibot-yellow-dark sm:text-sm"
                      />
                    </div>
                  ))
                )}
              </div>

              {error && (
                <div id="documentGenerationError" className="text-red-500 text-sm">
                  {error}
                </div>
              )}

              <div className="flex justify-end space-x-3">
                <button
                  type="button"
                  onClick={() => { setIsDocumentFillModalOpen(false); setError(null); }}
                  className="px-4 py-2 bg-gray-200 text-gray-800 rounded-md hover:bg-gray-300 focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-opacity-75"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  id="generateDocumentButton"
                  className="px-4 py-2 bg-lexibot-yellow-dark text-white rounded-md hover:bg-lexibot-yellow focus:outline-none focus:ring-2 focus:ring-lexibot-yellow-dark focus:ring-opacity-75 disabled:opacity-50"
                  disabled={isLoading}
                >
                  {isLoading ? 'Generating...' : 'Generate Document'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </>
  );
}

export default App;