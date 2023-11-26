from faster_whisper import WhisperModel
import time
import os

#preso in input un tipo "<class 'faster_whisper.transcribe.TranscriptionInfo'>"
#relativo a una traccia mp3, restituisce lingua, probabilità e durata della traccia 
def get_info(info):
    language=info.language
    language_probability=round(info.language_probability,3)
    duration=info.duration
    return language, language_probability, duration
    
#preso in input un valore in secondi, lo restituisce secondo la formattazione hh mm ss
def format_time(time_sec):
    h=int(time_sec/3600)
    time_sec-=h*3600
    m=int(time_sec/60)
    time_sec-=m*60
    s=int(time_sec)
    formatted_time=f"{h}h {m}m {s}s"
    return formatted_time
    
#data in input una lista di cartelle, se non esistono le genera appositamente
def generate_folder(paths_list):
    for path in paths_list:
        if os.path.isdir(path)==False:
            os.mkdir(path)

#presa in input una lista di opzioni, le stampa a schermo con un indice univoco
#consentendo di sceglierne una
def choose_from_list(options_list, message):
    print(message)
    for i in range(len(options_list)):
        print(f"\t[{i}] {options_list[i]}")
    print()
    n=int(input("Effettua una scelta: "))
    choice=options_list[n]
    return choice

#restituisce una lista dei file con un dato formato da un percorso dato in input 
def files_by_format(path, ext):
    ext_list=[]
    for cartelle, sottocartelle, files in os.walk(path):
        for file in files:
            if file.endswith(ext):
                ext_list.append(file)
    return ext_list
    
def export_text(path, name, string):
    file=open( os.path.join(path, f"{name}.txt"), "w" )
    file.write(string)
    file.close
#------------------------------------------------------------------------------

home_path=os.getcwd()
source_path=os.path.join(home_path, "source")
output_path=os.path.join(home_path, "output")

#genera le cartelle di funzionamento
generate_folder([source_path, output_path])

mp3_list=files_by_format(source_path, ".mp3")

source_file=choose_from_list(mp3_list, "FILE PRESENTI: ")
source_file_path=os.path.join( source_path, source_file )  #percorso del file
file_name=source_file.replace(".mp3", "")           #elimina estensione dal nome del file
project_path = os.path.join( output_path, file_name )   #percorso di output del progetto
generate_folder( [project_path] )   #genera la cartella di progetto

#PARAMETRI WHISPER
#model_size = "large-v3"
models_list=("tiny", "base", "small", "medium", "large", "large-v2", "large-v3")
model_size = choose_from_list(models_list, "DIMENSIONE DEL MODELLO:")
model = WhisperModel(model_size, compute_type="int8")

#apre 'source file'
start_moment=time.time()
segments, info = model.transcribe(
    source_file_path,
    beam_size=1,    #provare con beam_size fra 5 e 10
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=2000),
)

#legge e stampa info del file aperto
language, language_prob, duration = get_info(info)
duration=format_time(duration)
#print(language, language_prob, duration)
print(f"Hai scelto '{source_file}'\t[Lingua '{language}' ({language_prob*100}%), durata {duration}]\n")

ts_transcripted_text=""
no_ts_transcripted_text=""
c_transcripted_text=""

n_segments=0
for segment in segments:
    n_segments+=1       #n° segmenti
    start, end = format_time(segment.start), format_time(segment.end)         #tempi di inizio e fine segmento
    s_text=segment.text     #testo segmento (tipo stringa)
    print(f"[{start}]\t{s_text}")  #stampa a schermo la trascrizione del segmento corrente
    ts_transcripted_text+=f"[{start}]\t{s_text}\n"  #trascrizione con time stamp (output 1)
    no_ts_transcripted_text+=f"{s_text}\n"       #trascrizione senza time stamp  (output 2)
    c_transcripted_text+=f"{s_text}"       #trascrizione continua (output 3)

#ESPORTAZIONE DEI FILE
export_text(project_path, f"ts_{file_name}", ts_transcripted_text)
export_text(project_path, f"no_ts_{file_name}", no_ts_transcripted_text)
export_text(project_path, f"c_{file_name}", c_transcripted_text)

exc_time=moment=format_time(time.time()-start_moment)
print(f"\n\nFinito. {duration} trascritti in {exc_time}.")
