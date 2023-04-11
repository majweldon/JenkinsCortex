
import subprocess
subprocess.run(["pip", "install", "openai"])
subprocess.run(["pip", "install", "soundfile"])
subprocess.run(["pip","install","pydub"])

import os
import openai
from numpy import True_
import gradio as gr
import soundfile as sf
from pydub import AudioSegment

# Load your API key from an environment variable or secret management service
openai.api_key = os.environ.get('OPENAI_SECRET_KEY')

note_transcript = ""

def transcribe(audio, history_type):
  global note_transcript    
     
  history_type_map = {
      "History": "Weldon_Note_Format.txt",
      "Physical": "Weldon_PE_Note_Format.txt",
      "H+P": "Weldon_Full_Note_Format.txt",
      "Impression/Plan": "Weldon_Impression_Note_Format.txt",
      "Handover": "Weldon_Handover_Note_Format.txt",
      "Meds Only": "Medications.txt",
      "EMS": "EMS_Handover_Note_Format.txt",
      "Triage": "Triage_Note_Format.txt"
   }
    
  file_name = history_type_map.get(history_type, "Weldon_Full_Note_Format.txt")
  with open(f"Format_Library/{file_name}", "r") as f:
      role = f.read()

  messages = [{"role": "system", "content": role}]

  ###### Create Dialogue Transcript from Audio Recording and Append(via Whisper)
  # Load the audio file (from filepath)
  audio_data, samplerate = sf.read(audio)

  #### Massage .wav and save as .mp3
  #audio_data = audio_data.astype("float32")
  #audio_data = (audio_data * 32767).astype("int16")
  #audio_data = audio_data.mean(axis=1)
  sf.write("Audio_Files/test.wav", audio_data, samplerate, subtype='PCM_16')
  if not os.path.exists("Audio_Files/test.wav"):
      print("Error: Failed to create test.wav file")
      return
  sound = AudioSegment.from_wav("Audio_Files/test.wav")
  sound.export("Audio_Files/test.mp3", format="mp3")
  if not os.path.exists("Audio_Files/test.mp3"):
      print("Error: Failed to create test.mp3 file")
      return

  #Send file to Whisper for Transcription
  audio_file = open("Audio_Files/test.mp3", "rb")
  audio_transcript = openai.Audio.transcribe("whisper-1", audio_file)
  print(audio_transcript)
  messages.append({"role": "user", "content": audio_transcript["text"]})
  
  #Create Sample Dialogue Transcript from File (for debugging)
  #with open('Audio_Files/Test_Elbow.txt', 'r') as file:
  #  audio_transcript = file.read()
  #messages.append({"role": "user", "content": audio_transcript})
  

  ### Word and MB Count
  file_size = os.path.getsize("Audio_Files/test.mp3")
  mp3_megabytes = file_size / (1024 * 1024)
  mp3_megabytes = round(mp3_megabytes, 2)

  audio_transcript_words = audio_transcript["text"].split() # Use when using mic input
  #audio_transcript_words = audio_transcript.split() #Use when using file

  num_words = len(audio_transcript_words)


  #Ask OpenAI to create note transcript
  response = openai.ChatCompletion.create(model="gpt-3.5-turbo", temperature=0, messages=messages)
  note_transcript = (response["choices"][0]["message"]["content"])
   
  return [note_transcript, num_words,mp3_megabytes]

#Define Gradio Interface
my_inputs = [
    gr.Audio(source="microphone", type="filepath"),
    gr.Radio(["History","Physical", "H+P","Impression/Plan","Handover","EMS","Triage","Meds Only"], show_label=False),
]

ui = gr.Interface(fn=transcribe, 
                  inputs=my_inputs, 
                  outputs=[gr.Textbox(label="Your Note").style(show_copy_button=True),
                           gr.Number(label="Audio Word Count"),
                           gr.Number(label=".mp3 MB")])


ui.launch(share=False, debug=True)