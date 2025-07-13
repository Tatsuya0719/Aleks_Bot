// src/DocumentFillModal.tsx
import { useState, useEffect } from 'react'; // Removed 'React' from import

interface PlaceholderDetail {
  name: string;
  description: string;
}

interface DocumentFillModalProps {
  documentType: string;
  placeholders: PlaceholderDetail[];
  onClose: () => void;
  onSubmit: (data: { [key: string]: string }) => void;
  loading: boolean;
  error: string | null;
}

const DocumentFillModal: React.FC<DocumentFillModalProps> = ({
  documentType,
  placeholders,
  onClose,
  onSubmit,
  loading,
  error,
}) => {
  const [formData, setFormData] = useState<{ [key: string]: string }>({});

  // Initialize form data when placeholders change
  useEffect(() => {
    const initialData: { [key: string]: string } = {};
    placeholders.forEach((p) => {
      initialData[p.name] = ''; // Initialize with empty string
    });
    setFormData(initialData);
  }, [placeholders]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: value,
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formData);
  };

  if (!documentType || placeholders.length === 0) {
    return null; // Don't render if no document type or placeholders
  }

  return (
    <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-md max-h-[90vh] overflow-y-auto">
        <div className="flex justify-between items-center p-4 border-b">
          <h2 className="text-xl font-semibold text-gray-800">Fill Out {documentType} Template</h2>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-700 text-2xl font-bold">
            &times;
          </button>
        </div>
        <form onSubmit={handleSubmit} className="p-4 space-y-4">
          <p className="text-sm text-gray-600">Please provide the following details:</p>
          {placeholders.map((p) => (
            <div key={p.name}>
              <label htmlFor={p.name} className="block text-sm font-medium text-gray-700">
                {p.description} <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                id={p.name}
                name={p.name}
                value={formData[p.name] || ''}
                onChange={handleChange}
                required
                className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
              />
            </div>
          ))}

          {error && <p className="text-red-500 text-sm">{error}</p>}

          <div className="flex justify-end space-x-3">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 bg-gray-200 text-gray-800 rounded-md hover:bg-gray-300 focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-opacity-75"
              disabled={loading}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-75 disabled:opacity-50"
              disabled={loading}
            >
              {loading ? 'Generating...' : 'Generate Document'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default DocumentFillModal;