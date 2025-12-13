import React, { useState } from 'react'
import './ChatMessage.css'

function ChatMessage({ message }) {
  const [showSources, setShowSources] = useState(false)
  
  const formatTime = (date) => {
    return new Date(date).toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    })
  }

  const hasSources = (message.sources && message.sources.length > 0) || (message.chunks && message.chunks.length > 0)
  const hasChunks = message.chunks && message.chunks.length > 0

  return (
    <div className={`message ${message.sender}-message ${message.isError ? 'error-message' : ''}`}>
      <div className="message-content">
        <div className="message-text">{message.text}</div>
        <div className="message-meta">
          <div className="message-timestamp">
            {formatTime(message.timestamp)}
          </div>
        </div>
        {hasSources && (
          <div className="message-sources">
            <button 
              className="sources-toggle"
              onClick={() => setShowSources(!showSources)}
              aria-label="Toggle sources"
            >
              {showSources ? '▼' : '▶'} Sources 
              {message.sources && message.sources.length > 0 && ` (${message.sources.length} docs`}
              {hasChunks && `, ${message.chunks.length} chunks`}
              {message.sources && message.sources.length > 0 && ')'}
            </button>
            {showSources && (
              <div className="sources-list">
                {message.sources && message.sources.length > 0 && (
                  <div className="sources-section">
                    <h4 className="sources-section-title">Source Documents:</h4>
                    {message.sources.map((source, index) => (
                      <div key={index} className="source-item">
                        📄 {source}
                      </div>
                    ))}
                  </div>
                )}
                {hasChunks && (
                  <div className="chunks-section">
                    <h4 className="sources-section-title">Retrieved Chunks:</h4>
                    {message.chunks.map((chunk, index) => (
                      <div key={index} className="chunk-item">
                        <div className="chunk-header">
                          <span className="chunk-source">📄 {chunk.source}</span>
                          {chunk.score !== undefined && (
                            <span className="chunk-score">Score: {chunk.score.toFixed(3)}</span>
                          )}
                        </div>
                        <div className="chunk-content">{chunk.content}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default ChatMessage
