import os
import openai
import time
from numpy import True_
import gradio as gr
from gradio_rich_textbox import RichTextbox
import soundfile as sf
from pydub import AudioSegment

from openai import OpenAI

########## For creating a debug report
#import subprocess
#myGradioEnvironment = subprocess.run(['gradio','environment'], stdout=subprocess.PIPE)
#print(myGradioEnvironment.stdout.decode('utf-8'))

# Load API key from an environment variable
OPENAI_SECRET_KEY = os.environ.get("OPENAI_SECRET_KEY")
client = OpenAI(api_key = OPENAI_SECRET_KEY)



note_transcript = ""

def transcribe(audio, history_type):
  global note_transcript
  print(f"Received audio file path: {audio}")
     
  history_type_map = {
      "History": "Weldon_History_Format.txt",
      "Physical": "Weldon_PE_Note_Format.txt",
      "H+P": "Weldon_History_Physical_Format.txt",
      "Impression/Plan": "Weldon_Impression_Note_Format.txt",
      "Handover": "Weldon_Handover_Note_Format.txt",
      "Meds Only": "Medications.txt",
      "EMS": "EMS_Handover_Note_Format.txt",
      "Triage": "Triage_Note_Format.txt",
      "Full Visit": "Weldon_Full_Visit_Format_HTML.txt",
      "Psych": "Weldon_Psych_Format.txt",
      "SBAR": "SBAR.txt"
      
   }
  file_name = history_type_map.get(history_type, "Weldon_Full_Visit_Format.txt")
  with open(f"Format_Library/{file_name}", "r") as f:
    role = f.read()
  messages = [{"role": "system", "content": role}]


  ######################## Take Audio from Numpy Array
  #samplerate, audio_data = audio

        
  ######################## Read audio file, if using file
  max_attempts = 1
  attempt = 0
  audio_data = None
  samplerate = None
  while attempt < max_attempts:
      try:
          if audio is None:
              raise TypeError("Invalid file: None")
          audio_data, samplerate = sf.read(audio)
          break
      except (OSError, TypeError) as e:
          print(f"Attempt {attempt + 1} of {max_attempts} failed with error: {e}")
          attempt += 1
          time.sleep(3)
  else:
      print(f"###############Failed to open audio file after {max_attempts} attempts.##############")
      return  # Terminate the function or raise an exception if the file could not be opened


  ########## Cast as float 32, normalize
  #audio_data = audio_data.astype("float32")
  #audio_data = (audio_data * 32767).astype("int16")
  #audio_data = audio_data.mean(axis=1)

  ###################Code to convert .wav to .mp3 (if neccesary)
  sf.write("Audio_Files/test.wav", audio_data, samplerate, subtype='PCM_16')
  sound = AudioSegment.from_wav("Audio_Files/test.wav")
  sound.export("Audio_Files/test.mp3", format="mp3")

  sf.write("Audio_Files/test.mp3", audio_data, samplerate)
  
    
  ################  Send file to Whisper for Transcription
  audio_file = open("Audio_Files/test.mp3", "rb")
  
  max_attempts = 3
  attempt = 0
  while attempt < max_attempts:
      try:
          audio_transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
          break
      except openai.error.APIConnectionError as e:
          print(f"Attempt {attempt + 1} failed with error: {e}")
          attempt += 1
          time.sleep(3) # wait for 3 seconds before retrying
  else:
      print("Failed to transcribe audio after multiple attempts")  
    
  print(audio_transcript.text)
  messages.append({"role": "user", "content": audio_transcript.text})
  
  #Create Sample Dialogue Transcript from File (for debugging)
  #with open('Audio_Files/Test_Elbow.txt', 'r') as file:
  #  audio_transcript = file.read()
  #messages.append({"role": "user", "content": audio_transcript})
  

  ### Word and MB Count
  file_size = os.path.getsize("Audio_Files/test.mp3")
  mp3_megabytes = file_size / (1024 * 1024)
  mp3_megabytes = round(mp3_megabytes, 2)

  audio_transcript_words = audio_transcript.text.split() # Use when using mic input
  #audio_transcript_words = audio_transcript.split() #Use when using file

  num_words = len(audio_transcript_words)


  #Ask OpenAI to create note transcript
  response = client.chat.completions.create(model="gpt-4-turbo-preview", temperature=0, messages=messages)
  #response = client.chat.completions.create(model="gpt-3.5-turbo", temperature=0, messages=messages)
    
  note_transcript = response.choices[0].message.content
  print(note_transcript)
  note_transcript = note_transcript.replace('\n', '<br>')
  return [note_transcript, num_words, mp3_megabytes]

#Define Gradio Interface
my_inputs = [
    #gr.Audio(source="microphone", type="filepath"), #Gradio 3.48.0
    #gr.Audio(sources=["microphone"], type="filepath",format="wav"), #Gradio 4.x
    #gr.Audio(sources=["microphone"],type="numpy",editable="false"), #Gradio 4.x
    gr.Microphone(type="filepath",format="mp3"), #Gradio 4.x
    gr.Radio(["History","H+P","Impression/Plan","Full Visit","Handover","Psych","EMS","SBAR","Meds Only"], show_label=False),
]

ui = gr.Interface(fn=transcribe,
                  inputs=my_inputs, 
                  outputs=[RichTextbox(label="Your Note (gpt4-turbo-preview)", elem_id="htext"),
                           #gr.Textbox(label="Your Note (GPT 3.5 Turbo)", show_copy_button=True),
                           gr.Number(label="Audio Word Count"),
                           gr.Number(label=".mp3 MB")],
                  css="#htext span {white-space: pre}"
                 )


ui.launch(share=False, debug=True)