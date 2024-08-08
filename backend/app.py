
from flask import Flask, request, jsonify
import openai
import os
import time
from datetime import datetime
from werkzeug.utils import secure_filename
from pydub import AudioSegment
from flask_pymongo import PyMongo, ObjectId
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from config import config #contain mongodb user and password
from flask_cors import CORS
import librosa
import numpy

FOLDER_TO_UPLOAD = '/Users/khanhvynguyen/Documents/Flask_git/Conversational_Speech_Recognition_AI/Flask/audio_save'
allowed_extensions = {'wav','mp3'}


device = "cuda:0" if torch.cuda.is_available() else "cpu" #get gpu
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

app = Flask(__name__)
mongo_uri = config.MONGO_URI
if not mongo_uri:
    raise ValueError("MONGO_URI environment variable is not set")

print(f"Connecting to MongoDB URI: {mongo_uri}")
app.config["MONGO_URI"] = mongo_uri

CORS(app)

mongo = PyMongo(app)
try:
    db = mongo.db
    users_collection = db.users
    print("Successfully connected to the database")
except AttributeError as e:
    print("Failed to initialize MongoDB client. Check your MONGO_URI and MongoDB setup.")
    print(e)
    raise


# Load the whisper model and tokenizer
model_name2 = "openai/whisper-small" #change to whisper small to process the audio faster 
# Although it is whisper small, the performance is still good compare to whisper large

try:
    speechmodel = AutoModelForSpeechSeq2Seq.from_pretrained(model_name2,torch_dtype=torch_dtype, low_cpu_mem_usage=False, use_safetensors=True
)
    speechmodel.to(device)

    processor = AutoProcessor.from_pretrained(model_name2)
except OSError:
    print(f"Model not found locally. Downloading {model_name2}...")
    speechmodel = AutoModelForSpeechSeq2Seq.from_pretrained(model_name2)


# OpenAI API key
openai.api_key = "yourkeyhere"
model_name = "gpt-3.5-turbo"


# system prompt
system_prompt = {
    "role": "system",
    "content": "You are an  assistant"
}


app.config['UPLOAD_FOLDER'] = FOLDER_TO_UPLOAD

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


@app.route("/")
def home():
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "No user ID provided"}), 400
    
    user_session = users_collection.find_one({"user_id": user_id})
    if user_session:
        current_question_index = user_session.get("current_question_index", 0)
    else:
        current_question_index = 0
    
    if current_question_index < len(questions):
        return jsonify({
            "message": "Hi, this is an open-ended section. Thank you for participating in the survey. Let's resume.",
            "question": questions[current_question_index],
            "user_id": user_id
        })
    else:
        return jsonify({
            "message": "Thank you for completing the survey. You have answered all questions.",
            "user_id": user_id
        })




@app.route('/get', methods=['GET'])
def get():

    user_id = request.args.get('user_id')
    user_input = request.args.get('msg')
    typing_duration = request.args.get('typing_duration',type = int)
    confirm_translation = request.args.get('confirm_translation', type=bool)
    if not user_id:
        return jsonify({'error': 'No user ID provided'}), 400

    try:
        user_session = users_collection.find_one({"user_id": user_id})
        print(f"Retrieved user session: {user_session}")

        if not user_session:
            user_session = {
                "user_id": user_id,
                "current_question_index": 0,
                "chat_history": [],
                "skip_history": 0
            }
            users_collection.insert_one(user_session)
            print(f"Inserted new user session for: {user_id}")
        else:
            user_session["chat_history"] = user_session.get("chat_history", [])  # resume if it's the same ID
        
        current_question_index = user_session["current_question_index"]
        skip_history = user_session["skip_history"]

        response_data = {"message": ""}
        print(f"Initialized response_data: {response_data}")

        if user_input.lower() in ["eh", "meh", "nah", "mah", "can you repeat?", "repeat the question", "repeat"]:
            response_data["message"] = "Sure, let me repeat the question."
            response_data["next_question"] = questions[current_question_index]
            user_session['chat_history'].append({"role": "user", "content": user_input})
            user_session['chat_history'].append({"role": "assistant", "content": response_data["message"]})
            users_collection.update_one({"user_id": user_id}, {"$set": user_session})
            return jsonify(response_data)
        #when users using the record button
        if confirm_translation:
            if user_input.lower() in ["yes", "no"]: 
                if user_input.lower() == "yes":
                    if current_question_index < len(questions):
                        current_question_index +=1
                        response_data["next_question"] = questions[current_question_index]
                        user_session["current_question_index"] = current_question_index
                else:
                    response_data["message"] = "Please re-record your audio or type the correct transcript."
                    response_data["next_question"] = None
            else:
                user_session['chat_history'].append({"role": "user", "content": user_input})
                current_question_index = current_question_index
                user_session["current_question_index"] = current_question_index
                response_data["message"] = f"Your response: {user_input}. Next question {current_question_index+2}/{(len(questions))}:"
                if current_question_index < len(questions)-1: #because the index start with 0
                    current_question_index +=1
                    response_data["next_question"] = questions[current_question_index]
                    user_session["current_question_index"] = current_question_index
                #if current_question_index >= len(questions):
                else:
                    response_data["message"] = "Thank you for completing the survey. You have answered all questions."
                    response_data["next_question"] = None

            if response_data["next_question"]:
                user_session["chat_history"].append({"role": "assistant", "content": response_data["next_question"]})

            users_collection.update_one({"user_id": user_id}, {"$set": user_session})

            return jsonify(response_data)

        if user_input.lower() in ["skip", "skip the question"]:
            response_data["message"] = "Sure, here is the next question."
            skip_history += 1
            user_session['skip_history'] = skip_history

            if skip_history >= 3:
                response_data["message"] = "Sorry, you can only skip 3 questions."
                response_data["next_question"] = questions[current_question_index]
            else:
                if current_question_index < len(questions)-1:
                    current_question_index +=1
                    response_data["next_question"] = questions[current_question_index]
                else:
                    response_data["message"] = "Thank you for completing the survey. You have answered all questions."
                    response_data["next_question"] = None
            
            user_session['current_question_index'] = current_question_index
            user_session['chat_history'].append({"role": "user", "content": user_input})
            user_session['chat_history'].append({"role": "assistant", "content": response_data["message"]})
            users_collection.update_one({"user_id": user_id}, {"$set": user_session})
            return jsonify(response_data)
        # Calculate word count in user input
        words = user_input.split()
        word_count = len(words)
        print(typing_duration)
        if typing_duration <= 5000 and word_count > 50:
            response_data["message"] = "Please type your answer again"
            response_data["next_question"] = questions[current_question_index]
            user_session['chat_history'].append({"role": "user", "content": user_input})
            user_session['chat_history'].append({"role": "assistant", "content": response_data["message"]})
            users_collection.update_one({"user_id": user_id}, {"$set": user_session})
            return jsonify(response_data)
   
        if user_input.lower() in ["rephrase", "rephrase the question", "i don't understand", "please help"]:
            response_data["message"] = "Let me rephrase the question."
            response_data["next_question"] = rephrased_questions.get(questions[current_question_index], questions[current_question_index])
            user_session['chat_history'].append({"role": "user", "content": user_input})
            user_session['chat_history'].append({"role": "assistant", "content": response_data["message"]})
            users_collection.update_one({"user_id": user_id}, {"$set": user_session})
            return jsonify(response_data)
        



        # Check if the user is asking for help or starting a conversation
        print(f"User input for OpenAI: {user_input}")

        # If user ask for help or to talk
        if "help" in user_input.lower() or "talk" in user_input.lower() or "please" in user_input.lower() or "can you" in user_input.lower() or "die" in user_input.lower():
            conversation_input = [system_prompt] + user_session["chat_history"] + [{"role": "user", "content": user_input}]

            response = openai.ChatCompletion.create(
                model=model_name,
                messages=conversation_input,
                max_tokens=150
            )

            bot_response = response.choices[0].message['content'].strip()
            response_data["message"] = bot_response
            print(f"Bot response: {bot_response}")

            response_data["message"] = bot_response
            current_question_index += 1
            user_session["current_question_index"] = current_question_index
            response_data["next_question"] = questions[current_question_index] if current_question_index < len(questions) else None

        else:
            user_session['chat_history'].append({"role": "user", "content": user_input})
            current_question_index = current_question_index
            user_session["current_question_index"] = current_question_index
            response_data["message"] = f"Your response: {user_input}. Next question {current_question_index+2}/{(len(questions))}:"
            if current_question_index < len(questions)-1: #because the index start with 0
                current_question_index +=1
                response_data["next_question"] = questions[current_question_index]
                user_session["current_question_index"] = current_question_index
            #if current_question_index >= len(questions):
            else:
                response_data["message"] = "Thank you for completing the survey. You have answered all questions."
                response_data["next_question"] = None

        if response_data["next_question"]:
            user_session["chat_history"].append({"role": "assistant", "content": response_data["next_question"]})

        users_collection.update_one({"user_id": user_id}, {"$set": user_session})

        return jsonify(response_data)

    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({'error': 'An internal error occurred'}), 500
    

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return jsonify({'error':'no file'}), 400

    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        if not file_path.endswith('.wav'):
            sound = AudioSegment.from_file(file_path)
            file_path_wav = file_path.rsplit('.', 1)[0] + '.wav'
            sound.export(file_path_wav, format="wav")
            os.remove(file_path)
            file_path = file_path_wav

        audio, sr = librosa.load(file_path)
        inputs = processor(audio, return_tensors="pt")

        with torch.no_grad():
            predicted_ids1 = speechmodel.generate(inputs["input_features"])
            predicted_ids2 = speechmodel.generate(inputs["input_features"], forced_decoder_ids=processor.tokenizer.get_decoder_prompt_ids(language='en', task='translate'))
        
        transcription = processor.batch_decode(predicted_ids1, skip_special_tokens=True)[0]
        translation = processor.batch_decode(predicted_ids2, skip_special_tokens=True)[0]
        return jsonify({'transcript': transcription, 'translation': translation, 'confirmation_needed': True}), 200


    else:
        return jsonify({'error': 'File type not supported'}), 400



if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)




