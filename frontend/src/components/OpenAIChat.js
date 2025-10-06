import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Bot, User, Loader, Sparkles, FileText, HelpCircle } from 'lucide-react';
import { apiService } from '../services/apiService';

const OpenAIChat = ({ onClose }) => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [chatMode, setChatMode] = useState('question'); // 'question', 'insights', 'report'
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Add welcome message
    setMessages([{
      id: 1,
      type: 'bot',
      content: 'Hello! I\'m your AI marine biology assistant. I can help you understand shark habitat predictions, answer questions about the data, and generate insights. What would you like to know?',
      timestamp: new Date()
    }]);
  }, []);

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputValue.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      let response;
      
      switch (chatMode) {
        case 'insights':
          response = await apiService.generateInsights({
            prediction_data: { /* Add current prediction data */ },
            shark_data: { /* Add current shark data */ }
          });
          break;
        case 'report':
          response = await apiService.generateReport({
            analysis_data: {
              model_performance: { /* Add model performance data */ },
              dataset_stats: { /* Add dataset stats */ },
              feature_importance: { /* Add feature importance data */ },
              temporal_analysis: { /* Add temporal analysis data */ },
              species_distribution: { /* Add species distribution */ }
            }
          });
          break;
        default:
          response = await apiService.askQuestion({
            question: inputValue.trim(),
            context_data: { /* Add context data */ }
          });
      }

      const botMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: response.answer || response.insights || response.report || 'I apologize, but I couldn\'t process your request.',
        timestamp: new Date()
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: 'I apologize, but I encountered an error processing your request. Please try again.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const quickQuestions = [
    "What factors influence shark foraging behavior?",
    "How accurate is the habitat prediction model?",
    "Which species are most active in this area?",
    "What environmental conditions favor shark foraging?",
    "How does sea surface temperature affect shark behavior?"
  ];

  const handleQuickQuestion = (question) => {
    setInputValue(question);
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.9 }}
      className="fixed inset-0 bg-shark-900/80 backdrop-blur-sm flex items-center justify-center z-50 p-4"
    >
      <div className="glass-card rounded-xl w-full max-w-2xl h-[80vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-shark-700">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
              <Bot className="w-5 h-5 text-white" />
            </div>
            <div>
              <h3 className="text-white font-semibold">AI Assistant</h3>
              <p className="text-ocean-300 text-sm">Marine Biology & Habitat Prediction</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-shark-700/50 rounded-lg transition-colors"
          >
            <span className="text-ocean-400">×</span>
          </button>
        </div>

        {/* Mode Selector */}
        <div className="p-4 border-b border-shark-700">
          <div className="flex space-x-2">
            {[
              { id: 'question', label: 'Q&A', icon: HelpCircle },
              { id: 'insights', label: 'Insights', icon: Sparkles },
              { id: 'report', label: 'Report', icon: FileText }
            ].map((mode) => {
              const Icon = mode.icon;
              return (
                <button
                  key={mode.id}
                  onClick={() => setChatMode(mode.id)}
                  className={`flex items-center space-x-2 px-3 py-2 rounded-lg text-sm transition-colors ${
                    chatMode === mode.id
                      ? 'bg-ocean-600/20 text-ocean-400 border border-ocean-500/30'
                      : 'bg-shark-700/50 text-ocean-300 hover:bg-shark-600/50'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{mode.label}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map((message) => (
            <motion.div
              key={message.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div className={`flex items-start space-x-3 max-w-[80%] ${
                message.type === 'user' ? 'flex-row-reverse space-x-reverse' : ''
              }`}>
                <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                  message.type === 'user' 
                    ? 'bg-ocean-600' 
                    : 'bg-gradient-to-br from-purple-500 to-pink-500'
                }`}>
                  {message.type === 'user' ? (
                    <User className="w-4 h-4 text-white" />
                  ) : (
                    <Bot className="w-4 h-4 text-white" />
                  )}
                </div>
                <div className={`p-3 rounded-lg ${
                  message.type === 'user'
                    ? 'bg-ocean-600 text-white'
                    : 'bg-shark-700 text-ocean-100'
                }`}>
                  <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                  <p className="text-xs opacity-70 mt-1">
                    {message.timestamp.toLocaleTimeString()}
                  </p>
                </div>
              </div>
            </motion.div>
          ))}
          
          {isLoading && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex justify-start"
            >
              <div className="flex items-start space-x-3">
                <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-pink-500 rounded-full flex items-center justify-center">
                  <Bot className="w-4 h-4 text-white" />
                </div>
                <div className="bg-shark-700 p-3 rounded-lg">
                  <div className="flex items-center space-x-2">
                    <Loader className="w-4 h-4 animate-spin text-ocean-400" />
                    <span className="text-sm text-ocean-300">Thinking...</span>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Quick Questions */}
        {messages.length === 1 && (
          <div className="p-4 border-t border-shark-700">
            <p className="text-sm text-ocean-400 mb-3">Quick questions:</p>
            <div className="flex flex-wrap gap-2">
              {quickQuestions.map((question, index) => (
                <button
                  key={index}
                  onClick={() => handleQuickQuestion(question)}
                  className="px-3 py-1 bg-shark-700/50 hover:bg-shark-600/50 border border-shark-600 rounded-full text-xs text-ocean-300 hover:text-white transition-colors"
                >
                  {question}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Input */}
        <div className="p-4 border-t border-shark-700">
          <div className="flex space-x-3">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask about shark habitat predictions..."
              className="flex-1 p-3 bg-shark-700/50 border border-shark-600 rounded-lg text-white placeholder-ocean-400 focus:outline-none focus:border-ocean-500"
              disabled={isLoading}
            />
            <button
              onClick={handleSendMessage}
              disabled={!inputValue.trim() || isLoading}
              className="p-3 bg-ocean-600 hover:bg-ocean-700 disabled:bg-shark-700 disabled:text-shark-400 border border-ocean-500 rounded-lg text-white transition-colors"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default OpenAIChat;