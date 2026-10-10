import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import Sources from './Sources';
import './Stage1.css';

export default function Stage1({ responses }) {
  const [activeTab, setActiveTab] = useState(0);

  if (!responses || responses.length === 0) {
    return null;
  }

  return (
    <div className="stage stage1">
      <h3 className="stage-title">Stage 1: Individual Responses</h3>

      <div className="tabs">
        {responses.map((resp, index) => (
          <button
            key={index}
            className={`tab ${activeTab === index ? 'active' : ''}`}
            onClick={() => setActiveTab(index)}
          >
            {resp.model.split('/')[1] || resp.model}
          </button>
        ))}
      </div>

      <div className="tab-content">
        <div className="model-name">{responses[activeTab].model}</div>
        <div className="response-text markdown-content">
          <ReactMarkdown>{responses[activeTab].response}</ReactMarkdown>
        </div>
        {responses[activeTab].search_failed && (
          <div className="sources-warning">
            Web search failed for this model — it answered without searching
          </div>
        )}
        {/* Searches this model chose to run with the web_search tool */}
        <Sources
          key={activeTab}
          sources={responses[activeTab].sources}
          queries={(responses[activeTab].searches || []).map((s) => s.query)}
          defaultExpanded={false}
        />
      </div>
    </div>
  );
}
