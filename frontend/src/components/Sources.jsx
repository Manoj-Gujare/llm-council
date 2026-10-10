import { useState } from 'react';
import './Sources.css';

function getDomain(url) {
  try {
    return new URL(url).hostname.replace(/^www\./, '');
  } catch {
    return url;
  }
}

function formatDate(value) {
  if (!value) return null;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
}

/**
 * Numbered web sources with the queries that found them.
 * Used per model in Stage 1, and for older conversations that stored one shared pool.
 */
export default function Sources({ sources, queries = [], note, defaultExpanded = true }) {
  const [expanded, setExpanded] = useState(defaultExpanded);

  if (!sources?.length && !queries.length) {
    return null;
  }

  return (
    <div className="sources">
      <button
        type="button"
        className="sources-header"
        onClick={() => setExpanded(!expanded)}
        aria-expanded={expanded}
      >
        <span className="stage-title">
          Web Sources ({sources?.length || 0})
        </span>
        <span className="sources-toggle">{expanded ? 'Hide' : 'Show'}</span>
      </button>

      {expanded && (
        <div className="sources-body">
          {note && <p className="sources-note">{note}</p>}

          {queries.length > 0 && (
            <div className="query-chips">
              {queries.map((query, i) => (
                <span key={i} className="query-chip">
                  {query}
                </span>
              ))}
            </div>
          )}

          {sources?.length > 0 && (
            <ol className="source-list">
              {sources.map((source) => {
                const date = formatDate(source.published_date);
                return (
                  <li key={source.id} className="source-item">
                    <span className="source-id">[{source.id}]</span>
                    <div className="source-info">
                      <a
                        href={source.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="source-title"
                      >
                        {source.title}
                      </a>
                      <div className="source-meta">
                        <span>{getDomain(source.url)}</span>
                        {date && <span>{date}</span>}
                      </div>
                    </div>
                  </li>
                );
              })}
            </ol>
          )}
        </div>
      )}
    </div>
  );
}
