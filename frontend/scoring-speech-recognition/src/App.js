import React, { useState, useEffect } from 'react';
import { ReactMic } from 'react-mic';
import axios from 'axios';
import './App.css';



function App() {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [isRecording, setIsRecording] = useState(false);
    const [userId, setUserId] = useState('');
    const [transcript, setTranscript] = useState('');
    const [translation, setTranslation] = useState('');
    const [confirmationNeeded, setConfirmationNeeded] = useState(false);
    const [startTime, setStartTime] = useState(null);
    const [waitingForNextQuestion, setWaitingForNextQuestion] = useState(false);
  
    useEffect(() => {
      const urlParams = new URLSearchParams(window.location.search);
      const userId = urlParams.get('user_id');
      setUserId(userId);
  
      // Fetch the initial message
      const fetchInitialMessage = async () => {
        try {
          const response = await axios.get(`http://127.0.0.1:5001/?user_id=${userId}`);
          const greetingMessage = {
            sender: 'bot',
            text: response.data.message,
          };
          const questionMessage = {
            sender: 'bot',
            text: response.data.question,
          };
          setMessages([greetingMessage, questionMessage]);
        } catch (error) {
          console.error('Error fetching initial message:', error);
        }
      };
  
      fetchInitialMessage();
    }, []);
  
    const handleSend = async () => {
      if (!input.trim()) return;
  
      const typingEndTime = Date.now();
      const typingDuration = typingEndTime - startTime;
  
      const userMessage = {
        sender: 'user',
        text: input,
      };
  
      setMessages((prevMessages) => [...prevMessages, userMessage]);
  
      try {
        let response;
        if (waitingForNextQuestion) {
          response = await axios.get(`http://127.0.0.1:5001/get?user_id=${userId}&msg=${encodeURIComponent(input)}&typing_duration=${typingDuration}&confirm_translation=true`);
          setWaitingForNextQuestion(false); //wait for response 
        } else {
          response = await axios.get(`http://127.0.0.1:5001/get?user_id=${userId}&msg=${encodeURIComponent(input)}&typing_duration=${typingDuration}`);
        }
        const responseText = response.data;
  
        if (responseText.next_question) {
          const botResponseMessage = {
            sender: 'bot',
            text: responseText.message,
          };
          const nextQuestionMessage = {
            sender: 'bot',
            text: responseText.next_question,
          };
          setMessages((prevMessages) => [...prevMessages, botResponseMessage, nextQuestionMessage]);
        } else {
          const botResponseMessage = {
            sender: 'bot',
            text: "Thank you for completing the survey.",
          };
          setMessages((prevMessages) => [...prevMessages, botResponseMessage]);
        }
      } catch (error) {
        console.error('Error sending message to backend:', error);
        const errorMessage = {
          sender: 'bot',
          text: 'An error occurred. Please try again later.',
        };
        setMessages((prevMessages) => [...prevMessages, errorMessage]);
      } finally {
        setInput(''); 
        setStartTime(null);
      }
    };
  
    const toggleRecording = () => {
      setIsRecording((prevState) => !prevState);
    };
  
    const onData = (recordedBlob) => {
      console.log('OnData: ', recordedBlob);
    };
  
    const onStop = async (recordedBlob) => {
      const formData = new FormData();
      formData.append('file', recordedBlob.blob, 'recording.wav');
  
      try {
        const response = await axios.post('http://127.0.0.1:5001/upload_audio', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });
  
        setTranscript(response.data.transcript);
        setTranslation(response.data.translation);
        setConfirmationNeeded(response.data.confirmation_needed);
      } catch (error) {
        console.error('Error with audio recording:', error);
      } finally {
        setIsRecording(false);
      }
    };
  
    const handleInputChange = (e) => {
      if (startTime === null) {
        setStartTime(Date.now());
      }
      setInput(e.target.value);
    };
  
    const handleKeyPress = (e) => {
      if (e.key === 'Enter') {
        handleSend();
      }
    };
  
    const handleSendClick = () => {
      handleSend();
    };
  
    const handleConfirmation = async (isCorrect) => {
      if (isCorrect) {
        const inputText = `Transcript: ${transcript}| Translation: ${translation}`;
        setInput(inputText);
        setWaitingForNextQuestion(true); 
      } else {
        setMessages((prevMessages) => [
          ...prevMessages,
          { sender: 'bot', text: 'Please re-record your audio or type the correct transcript.' },
        ]);
      }
  
      setConfirmationNeeded(false);
      setTranscript('');
      setTranslation('');
    };
  
    return (
      <div className="App">
        <div className="chat-container">
          <div className="logo-container">
            <img src={logo} width={200} height={200} alt="logo" className="logo" />
          </div>
          <h1>Your Survey Chat Bot</h1>
          <div className="messages">
            {messages.map((message, index) => (
              <div
                key={index}
                className={`message ${message.sender === 'user' ? 'user-message' : 'bot-message'}`}
              >
                {message.text}
              </div>
            ))}
            {transcript && (
              <div className="message bot-message">
                <p><strong>Transcript:</strong> {transcript}</p>
                <p><strong>Translation:</strong> {translation}</p>
                {confirmationNeeded && (
                  <div>
                    <p>Is this translation correct?</p>
                    <button onClick={() => handleConfirmation(true)}>Yes</button>
                    <button onClick={() => handleConfirmation(false)}>No</button>
                  </div>
                )}
              </div>
            )}
          </div>
          <div className="input-container">
            <input
              type="text"
              value={input}
              onChange={handleInputChange}
              placeholder="Type your message..."
              onKeyPress={handleKeyPress}
            />
            <button onClick={handleSendClick}>Send</button>
            <button onClick={toggleRecording}>
              {isRecording ? "Stop Recording" : "Start Recording"}
            </button>
            <ReactMic
              record={isRecording}
              className={`sound-wave ${isRecording ? '' : 'hideSoundWave'}`}
              onStop={onStop}
              onData={onData}
              strokeColor="#000000"
            />
          </div>
        </div>
      </div>
    );
  }
  
  export default App;
  
  