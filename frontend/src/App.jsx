import React, { useState, useRef, useEffect } from 'react'
import ChatMessage from './components/ChatMessage'
import ChatInput from './components/ChatInput'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || (import.meta.env.PROD ? '' : 'http://localhost:8000')

const SAMPLE_QUERIES = [
  "How do I reset my password?",
  "What is the vacation policy?",
  "How do I set up VPN?"
]

function App() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Hello! How can I help you today? You can ask me any questions you have about the knowledge base.",
      sender: 'bot',
      timestamp: new Date()
    }
  ])
  const [isLoading, setIsLoading] = useState(false)
  const [sessionId, setSessionId] = useState(null)
  const [showUpload, setShowUpload] = useState(false)
  const [showViewKB, setShowViewKB] = useState(false)
  const [showAnalytics, setShowAnalytics] = useState(false)
  const [analytics, setAnalytics] = useState(null)
  const [kbFiles, setKbFiles] = useState([])
  const [selectedFile, setSelectedFile] = useState(null)
  const [fileContent, setFileContent] = useState(null)
  const [isUploading, setIsUploading] = useState(false)
  const messagesEndRef = useRef(null)

  useEffect(() => {
    createSession()
    loadAnalytics()
    loadKBFiles()
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const createSession = async () => {
    try {
      const response = await fetch(`${API_URL}/sessions`, {
        method: 'POST'
      })
      if (response.ok) {
        const data = await response.json()
        setSessionId(data.session_id)
      }
    } catch (error) {
      console.error('Error creating session:', error)
    }
  }

  const loadAnalytics = async () => {
    try {
      const response = await fetch(`${API_URL}/analytics`)
      if (response.ok) {
        const data = await response.json()
        setAnalytics(data)
      }
    } catch (error) {
      console.error('Error loading analytics:', error)
    }
  }

  const loadKBFiles = async () => {
    try {
      const url = sessionId 
        ? `${API_URL}/knowledge-base/files?session_id=${encodeURIComponent(sessionId)}`
        : `${API_URL}/knowledge-base/files`
      const response = await fetch(url)
      if (response.ok) {
        const data = await response.json()
        setKbFiles(data.files || [])
      }
    } catch (error) {
      console.error('Error loading KB files:', error)
    }
  }

  const loadFileContent = async (filePath) => {
    try {
      const response = await fetch(`${API_URL}/knowledge-base/files/${encodeURIComponent(filePath)}`)
      if (response.ok) {
        const data = await response.json()
        setFileContent(data)
        setSelectedFile(filePath)
      } else {
        alert('Error loading file content')
      }
    } catch (error) {
      console.error('Error loading file content:', error)
      alert('Error loading file content')
    }
  }

  const deleteFile = async (filePath) => {
    if (!sessionId) {
      alert('Session ID is required to delete files')
      return
    }
    
    if (!confirm(`Are you sure you want to delete ${filePath}?`)) {
      return
    }
    
    try {
      const url = `${API_URL}/knowledge-base/files/${encodeURIComponent(filePath)}?session_id=${encodeURIComponent(sessionId)}`
      const response = await fetch(url, {
        method: 'DELETE'
      })
      if (response.ok) {
        alert('File deleted successfully')
        await loadKBFiles()
        if (selectedFile === filePath) {
          setFileContent(null)
          setSelectedFile(null)
        }
      } else {
        const error = await response.json()
        alert(`Error: ${error.detail || 'Failed to delete file'}`)
      }
    } catch (error) {
      console.error('Delete error:', error)
      alert('Error deleting file. Please try again.')
    }
  }

  const handleSampleQuery = (query) => {
    handleSendMessage(query)
  }

  const handleSendMessage = async (message) => {
    if (!message.trim() || isLoading) return

    const userMessage = {
      id: Date.now(),
      text: message,
      sender: 'user',
      timestamp: new Date()
    }
    setMessages(prev => [...prev, userMessage])
    setIsLoading(true)

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: message,
          session_id: sessionId,
          max_tokens: 1000,
          min_score: 0.5
        })
      })

      if (!response.ok) {
        throw new Error('Failed to get response')
      }

      const data = await response.json()
      
      if (data.session_id && data.session_id !== sessionId) {
        setSessionId(data.session_id)
      }
      
      const botMessage = {
        id: Date.now() + 1,
        text: data.response,
        sender: 'bot',
        timestamp: new Date(),
        intent: data.intent,
        confidence: data.confidence,
        modelUsed: data.model_used,
        sources: data.sources || [],
        chunks: data.chunks || []
      }
      setMessages(prev => [...prev, botMessage])
      
      loadAnalytics()
    } catch (error) {
      console.error('Error:', error)
      const errorMessage = {
        id: Date.now() + 1,
        text: "Sorry, I encountered an error. Please try again.",
        sender: 'bot',
        timestamp: new Date(),
        isError: true
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleClearChat = async () => {
    setMessages([
      {
        id: 1,
        text: "Hello! How can I help you today? You can ask me any questions you have about the knowledge base.",
        sender: 'bot',
        timestamp: new Date()
      }
    ])
    await createSession()
  }

  const handleFileUpload = async (files) => {
    if (!files || files.length === 0) return
    
    const formData = new FormData()
    Array.from(files).forEach(file => {
      formData.append('files', file)
    })
    if (sessionId) {
      formData.append('session_id', sessionId)
    }

    setIsUploading(true)
    try {
      const response = await fetch(`${API_URL}/knowledge-base/upload`, {
        method: 'POST',
        body: formData
      })

      if (response.ok) {
        const data = await response.json()
        alert(`Successfully uploaded ${data.file_count} file(s) to your session`)
        setShowUpload(false)
        loadAnalytics()
        loadKBFiles()
      } else {
        const error = await response.json()
        alert(`Error: ${error.detail || 'Failed to upload files'}`)
      }
    } catch (error) {
      console.error('Upload error:', error)
      alert('Error uploading files. Please try again.')
    } finally {
      setIsUploading(false)
    }
  }

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i]
  }

  return (
    <div className="app">
      <div className="chat-container">
        <div className="chat-header">
          <div className="header-content">
            <h1 
              style={{ cursor: 'pointer' }}
              onClick={() => {
                setShowUpload(false)
                setShowViewKB(false)
                setShowAnalytics(false)
                setFileContent(null)
                setSelectedFile(null)
              }}
            >
              Knowledge Base Assistant
            </h1>
          </div>
          <div className="header-actions">
            <button 
              className="upload-button"
              onClick={() => {
                setShowViewKB(false)
                setShowUpload(!showUpload)
              }}
            >
              View or Upload KB
            </button>
            <button 
              className="analytics-button"
              onClick={() => {
                setShowAnalytics(!showAnalytics)
                loadAnalytics()
              }}
            >
               Analytics
            </button>
            <button className="clear-button" onClick={handleClearChat}>
              Clear Chat
            </button>
          </div>
        </div>

        {(showUpload || showViewKB) && (
          <div className="upload-panel">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <div className="kb-panel-tabs" style={{ flex: 1 }}>
                <button 
                  className={showViewKB ? '' : 'active'}
                  onClick={() => {
                    setShowViewKB(false)
                    setShowUpload(true)
                  }}
                >
                  Upload
                </button>
                <button 
                  className={showViewKB ? 'active' : ''}
                  onClick={async () => {
                    setShowUpload(false)
                    setShowViewKB(true)
                    await loadKBFiles()
                  }}
                >
                  View Files
                </button>
              </div>
              <button 
                onClick={() => {
                  setShowUpload(false)
                  setShowViewKB(false)
                  setFileContent(null)
                  setSelectedFile(null)
                }}
                style={{
                  background: 'none',
                  border: 'none',
                  fontSize: '24px',
                  cursor: 'pointer',
                  color: '#666',
                  padding: '4px 8px',
                  marginLeft: '16px'
                }}
                title="Close"
              >
                ×
              </button>
            </div>

            {showUpload && (
              <div>
                <h3>Upload Knowledge Base</h3>
                <p>Upload files (.txt, .md, .pdf) or a .zip folder containing multiple files</p>
                <input
                  type="file"
                  accept=".txt,.md,.pdf,.zip"
                  multiple
                  disabled={isUploading}
                  onChange={(e) => {
                    if (e.target.files && e.target.files.length > 0) {
                      handleFileUpload(e.target.files)
                    }
                  }}
                />
                {isUploading && (
                  <div style={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    gap: '12px', 
                    marginTop: '16px',
                    color: '#667eea',
                    fontWeight: '500'
                  }}>
                    <div className="typing-indicator" style={{ 
                      display: 'flex', 
                      gap: '4px', 
                      padding: '8px 0'
                    }}>
                      <span style={{
                        width: '8px',
                        height: '8px',
                        borderRadius: '50%',
                        background: '#667eea',
                        animation: 'typing 1.4s infinite'
                      }}></span>
                      <span style={{
                        width: '8px',
                        height: '8px',
                        borderRadius: '50%',
                        background: '#667eea',
                        animation: 'typing 1.4s infinite',
                        animationDelay: '0.2s'
                      }}></span>
                      <span style={{
                        width: '8px',
                        height: '8px',
                        borderRadius: '50%',
                        background: '#667eea',
                        animation: 'typing 1.4s infinite',
                        animationDelay: '0.4s'
                      }}></span>
                    </div>
                    <span>Uploading files...</span>
                  </div>
                )}
              </div>
            )}

            {showViewKB && (
              <div>
                <h3>Knowledge Base Files</h3>
                {kbFiles.length === 0 ? (
                  <p>No files in knowledge base</p>
                ) : (
                  <div className="kb-view-container">
                    <div className="kb-files-list">
                      <p>Total files: {kbFiles.length}</p>
                      <ul>
                        {kbFiles.map((file, idx) => (
                          <li 
                            key={idx}
                            className={selectedFile === file.path ? 'selected' : ''}
                          >
                            <div 
                              style={{ flex: 1, cursor: 'pointer' }}
                              onClick={() => loadFileContent(file.path)}
                            >
                              <span className="file-name">{file.filename}</span>
                              <span className="file-info">
                                {file.type} • {formatFileSize(file.size)}
                                {file.is_session_file && ' • (Your file)'}
                              </span>
                            </div>
                            {file.is_session_file && (
                              <button
                                onClick={(e) => {
                                  e.stopPropagation()
                                  deleteFile(file.path)
                                }}
                                style={{
                                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                                  color: 'white',
                                  border: 'none',
                                  borderRadius: '4px',
                                  padding: '4px 8px',
                                  cursor: 'pointer',
                                  fontSize: '12px',
                                  marginLeft: '8px',
                                  transition: 'all 0.2s'
                                }}
                                onMouseEnter={(e) => {
                                  e.target.style.opacity = '0.9'
                                  e.target.style.transform = 'translateY(-1px)'
                                }}
                                onMouseLeave={(e) => {
                                  e.target.style.opacity = '1'
                                  e.target.style.transform = 'translateY(0)'
                                }}
                                title="Delete file"
                              >
                                Delete
                              </button>
                            )}
                          </li>
                        ))}
                      </ul>
                    </div>
                    {fileContent && (
                      <div className="kb-file-content">
                        <div className="file-content-header">
                          <h4>{fileContent.filename}</h4>
                          <button onClick={() => {
                            setFileContent(null)
                            setSelectedFile(null)
                          }}>✕</button>
                        </div>
                        <div className="file-content-body">
                          <pre>{fileContent.content}</pre>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {showAnalytics && analytics && (
          <div className="analytics-panel">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <h3 style={{ margin: 0 }}>Analytics</h3>
              <button 
                onClick={() => setShowAnalytics(false)}
                style={{
                  background: 'none',
                  border: 'none',
                  fontSize: '24px',
                  cursor: 'pointer',
                  color: '#666',
                  padding: '4px 8px'
                }}
                title="Close"
              >
                ×
              </button>
            </div>
            <div className="analytics-grid">
              <div className="stat-card">
                <h4>Total Sessions</h4>
                <p>{analytics.total_sessions}</p>
              </div>
              <div className="stat-card">
                <h4>Total Messages</h4>
                <p>{analytics.total_messages}</p>
              </div>
              <div className="stat-card">
                <h4>Sessions Today</h4>
                <p>{analytics.sessions_today}</p>
              </div>
              <div className="stat-card">
                <h4>Messages Today</h4>
                <p>{analytics.messages_today}</p>
              </div>
            </div>
            <div className="popular-queries">
              <h4>Popular Queries</h4>
              <ul>
                {analytics.popular_queries.slice(0, 5).map((item, idx) => (
                  <li key={idx}>
                    <span className="query-text">{item.query}</span>
                    <span className="query-count">{item.count}x</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}
        
        <div className="messages-container">
          {messages.length === 1 && (
            <div className="sample-queries">
              <h3>Try asking:</h3>
              <div className="sample-queries-grid">
                {SAMPLE_QUERIES.map((query, idx) => (
                  <button
                    key={idx}
                    className="sample-query-button"
                    onClick={() => handleSampleQuery(query)}
                    disabled={isLoading}
                  >
                    {query}
                  </button>
                ))}
              </div>
            </div>
          )}
          {messages.map((message) => (
            <ChatMessage key={message.id} message={message} />
          ))}
          {isLoading && (
            <div className="message bot-message">
              <div className="message-content">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <ChatInput onSendMessage={handleSendMessage} disabled={isLoading} />
      </div>
    </div>
  )
}

export default App
