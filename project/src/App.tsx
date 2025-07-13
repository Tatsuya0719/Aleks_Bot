// src/App.tsx
import { useState, useEffect, useRef } from 'react'; // Removed 'React' from import
import DocumentFillModal from './DocumentFillModal'; // Import the new modal component

interface Message {
  sender: 'user' | 'aleks';
  text: string;
  type: 'text' | 'document_request' | 'document_generated';
  documentType?: string;
  placeholdersToFill?: { name: string; description: string }[];
  generatedDocumentPreview?: string;
  sources?: { source: string; startIndex: string; snippet: string }[];
}

function App() {
  const [messages, setMessages] = useState<Message[]>([
    { sender: 'aleks', text: 'Hi there! I am Aleks, your AI legal assistant. How can I help you today?', type: 'text' },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // State for document filling modal
  const [showDocumentModal, setShowDocumentModal] = useState(false);
  const [currentDocumentType, setCurrentDocumentType] = useState<string>('');
  const [currentPlaceholders, setCurrentPlaceholders] = useState<
    { name: string; description: string }[]
  >([]);
  const [documentGenerationError, setDocumentGenerationError] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const API_BASE_URL = 'http://localhost:8000'; // Make sure this matches your FastAPI server port

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async (e: React.FormEvent | React.KeyboardEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage: Message = { sender: 'user', text: input, type: 'text' };
    setMessages((prevMessages) => [...prevMessages, userMessage]);
    setInput('');
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: input }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'An unknown error occurred.');
      }

      const data = await response.json();
      if (data.type === 'document_request') {
        setMessages((prevMessages) => [
          ...prevMessages,
          {
            sender: 'aleks',
            text: data.message,
            type: 'document_request',
            documentType: data.document_type,
            placeholdersToFill: data.placeholders_to_fill,
          },
        ]);
        setCurrentDocumentType(data.document_type);
        setCurrentPlaceholders(data.placeholders_to_fill);
        setShowDocumentModal(true);
      } else if (data.type === 'rag_response') {
        setMessages((prevMessages) => [
          ...prevMessages,
          { sender: 'aleks', text: data.response, type: 'text', sources: data.sources },
        ]);
      } else if (data.type === 'text') {
        setMessages((prevMessages) => [
          ...prevMessages,
          { sender: 'aleks', text: data.response, type: 'text' },
        ]);
      }
    } catch (err) {
      console.error('Error sending message:', err);
      setError((err as Error).message);
      setMessages((prevMessages) => [
        ...prevMessages,
        {
          sender: 'aleks',
          text: `Error: ${
            (err as Error).message
          }. Please ensure the Aleks AI API server is running.`,
          type: 'text',
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleDocumentSubmit = async (formData: { [key: string]: string }) => {
    setLoading(true);
    setDocumentGenerationError(null);

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
      setMessages((prevMessages) => [
        ...prevMessages,
        {
          sender: 'aleks',
          text: data.message,
          type: 'document_generated',
          generatedDocumentPreview: data.generated_document_preview,
        },
      ]);
      setShowDocumentModal(false); // Close modal on success
      setCurrentDocumentType('');
      setCurrentPlaceholders([]);
    } catch (err) {
      console.error('Error generating document:', err);
      setDocumentGenerationError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const handleModalClose = () => {
    setShowDocumentModal(false);
    setCurrentDocumentType('');
    setCurrentPlaceholders([]);
    setDocumentGenerationError(null);
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col">
      {/* Header */}
      <header className="bg-blue-600 text-white p-4 shadow-md text-center">
        <h1 className="text-2xl font-bold">Aleks PH Legal Assistant</h1>
      </header>

      {/* Chat Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-md px-4 py-2 rounded-lg shadow ${
                msg.sender === 'user'
                  ? 'bg-blue-500 text-white'
                  : 'bg-white text-gray-800 border border-gray-200'
              }`}
            >
              <p className="font-semibold mb-1">
                {msg.sender === 'user' ? 'You' : 'Aleks'}
              </p>
              <div className="whitespace-pre-wrap">{msg.text}</div>

              {/* Display Sources for RAG responses */}
              {msg.type === 'text' && msg.sources && msg.sources.length > 0 && (
                <div className="mt-2 text-xs text-gray-600 border-t pt-2">
                  <p className="font-medium">Sources:</p>
                  <ul className="list-disc pl-4">
                    {msg.sources.map((source, sIdx) => (
                      <li key={sIdx}>
                        <strong>{source.source}</strong> (Start Index: {source.startIndex})
                        <p className="italic text-gray-500 line-clamp-2">"{source.snippet}"</p>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Display generated document preview */}
              {msg.type === 'document_generated' && msg.generatedDocumentPreview && (
                <div className="mt-2 text-sm bg-gray-50 p-3 rounded-md border border-gray-200 overflow-x-auto">
                  <p className="font-semibold mb-1">Generated Document Preview:</p>
                  <pre className="whitespace-pre-wrap text-gray-700 text-xs font-mono">
                    {msg.generatedDocumentPreview}
                  </pre>
                  <p className="text-xs text-gray-500 mt-1">
                    (Document saved in your Python server's `document_templates` folder)
                  </p>
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <form onSubmit={sendMessage} className="bg-white p-4 border-t flex items-center shadow-lg">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={loading ? 'Thinking...' : 'Type your message...'}
          className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          disabled={loading || showDocumentModal} // Disable input when loading or modal is open
        />
        <button
          type="submit"
          className="ml-4 px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
          disabled={loading || showDocumentModal} // Disable button when loading or modal is open
        >
          {loading ? 'Sending...' : 'Send'}
        </button>
      </form>

      {/* Error Display */}
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mx-4 mb-4">
          <strong className="font-bold">Error:</strong>
          <span className="block sm:inline ml-2">{error}</span>
        </div>
      )}

      {/* Document Fill Modal */}
      {showDocumentModal && (
        <DocumentFillModal
          documentType={currentDocumentType}
          placeholders={currentPlaceholders}
          onClose={handleModalClose}
          onSubmit={handleDocumentSubmit}
          loading={loading}
          error={documentGenerationError}
        />
      )}
    </div>
  );
}

export default App;