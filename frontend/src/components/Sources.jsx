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

function shortModel(model) {
  return model.split('/')[1] || model;
}

export default function Sources({ search }) {
  const [expanded, setExpanded] = useState(true);

  if (!search) {
    return null;
  }

  if (search.status === 'failed') {
    return (
      <div className="sources-warning">
        Web search failed — council answered without sources
      </div>
    );
  }

  if (search.status !== 'ok' || !search.sources?.length) {
    return null;
  }

  const gapQueries = search.gap_queries || [];

  return (
    <div className="stage sources">
      <button
        type="button"
        className="sources-header"
        onClick={() => setExpanded(!expanded)}
        aria-expanded={expanded}
      >
        <span className="stage-title">
          Web Sources ({search.sources.length})
        </span>
        <span className="sources-toggle">{expanded ? 'Hide' : 'Show'}</span>
      </button>

      {expanded && (
        <div className="sources-body">
          <p className="sources-note">
            Every council member received these exact sources.
          </p>

          <div className="query-chips">
            {search.queries.map((query, i) => (
              <span key={`q-${i}`} className="query-chip">
                {query}
              </span>
            ))}
            {gapQueries.map((gap, i) => (
              <span
                key={`g-${i}`}
                className="query-chip gap"
                title={`Gap-fill, requested by: ${gap.requested_by
                  .map(shortModel)
                  .join(', ')}`}
              >
                + {gap.query}
              </span>
            ))}
          </div>

          <ol className="source-list">
            {search.sources.map((source) => {
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
                      {source.gap && <span className="source-gap">gap-fill</span>}
                    </div>
                  </div>
                </li>
              );
            })}
          </ol>
        </div>
      )}
    </div>
  );
}
