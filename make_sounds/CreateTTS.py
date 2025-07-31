from gtts import gTTS
import os

def create_tts_audio(text, lang="zh-tw", filename=None, output_dir="resources/sounds"):

    if filename is None:
        filename = f"{text}.mp3"

    os.makedirs(output_dir, exist_ok=True)
   
    output_path = os.path.join(output_dir, filename)
    

    tts = gTTS(text=text, lang=lang)
    tts.save(output_path)
    
    print(f"saved at: {output_path}")
    return output_path

squat_correct_texts = {
    "squat_correct": "深蹲姿勢正確，請保持這個姿勢",
    # "squat_good_form": "很好！深蹲姿勢很標準",
    # "squat_perfect": "完美的深蹲姿勢，繼續保持",
    # "squat_maintain": "姿勢正確，請維持這個動作",
    # "squat_well_done": "做得好！深蹲姿勢很到位",
    # "squat_keep_going": "姿勢正確，請繼續",
    # "squat_excellent": "優秀的深蹲姿勢",
    # "squat_continue": "姿勢很好，請繼續保持"
}


for key, text in squat_correct_texts.items():
    create_tts_audio(text, "zh-tw", f"{key}.mp3")

print("Created audios!")
