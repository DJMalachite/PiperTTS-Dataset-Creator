import os
import sys
import shutil
import ffmpeg
import whisper
from pydub import AudioSegment
from pydub.silence import split_on_silence

# ========== Configuration ==========
INPUT_EXTENSION = ".wav"  # change to ".wav" or ".mp3" as needed
MIN_SILENCE_LEN = 500     # in ms
SILENCE_THRESH = -40      # in dBFS
KEEP_SILENCE = 250        # in ms
WHISPER_MODEL_SIZE = "turbo" # Set the OpenAI Whisper model, examples can be found here: https://github.com/openai/whisper?tab=readme-ov-file#available-models-and-languages
WHISPER_DEVICE = "cuda" # Set between CPU or CUDA for GPU accelleration
# ===================================

def check_ffmpeg_available():
    from shutil import which
    if which("ffmpeg") is None:
        print("❌ ffmpeg not found in system path.")
        sys.exit(1)

def split_audio(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    file_counter = 1
    all_successful = True

    print("\n🔍 Step 1: Splitting audio by silence...")
    for filename in sorted(os.listdir(input_folder)):
        if not filename.lower().endswith(INPUT_EXTENSION):
            continue

        filepath = os.path.join(input_folder, filename)
        print(f"🔊 Processing {filename}...")

        try:
            # Use pydub to load based on extension
            audio = AudioSegment.from_file(filepath)

            chunks = split_on_silence(
                audio,
                min_silence_len=MIN_SILENCE_LEN,
                silence_thresh=SILENCE_THRESH,
                keep_silence=KEEP_SILENCE
            )

            if not chunks:
                print(f"⚠️ No silence found in {filename}. Skipping.")
                continue

            for chunk in chunks:
                output_path = os.path.join(output_folder, f"{file_counter}.wav")
                chunk.export(output_path, format="wav")
                print(f"✅ Saved: {output_path}")
                file_counter += 1

            # Delete the original file after successful split
            os.remove(filepath)
            print(f"🗑️ Deleted original input file: {filename}")

        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")
            all_successful = False

    return all_successful

def convert_audio(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    wav_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.wav')]

    if not wav_files:
        print(f"⚠️ No WAV files to convert in '{input_folder}'")
        return False

    all_successful = True

    print("\n🎛 Step 2: Converting audio format...")
    for count, wav_file in enumerate(wav_files, start=1):
        input_path = os.path.join(input_folder, wav_file)
        output_file = f"{count}.wav"
        output_path = os.path.join(output_folder, output_file)

        print(f"🎵 Converting: {wav_file} -> {output_file}")
        try:
            (
                ffmpeg
                .input(input_path)
                .output(output_path, acodec='pcm_s16le', ar=22050)
                .run(quiet=True, overwrite_output=True)
            )
        except ffmpeg.Error as e:
            print(f"❌ Conversion failed for {wav_file}: {e.stderr.decode() if e.stderr else e}")
            all_successful = False

    return all_successful

def transcribe_audio(input_folder, output_csv):
    model = whisper.load_model(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE)

    print("\n🧠 Step 3: Transcribing audio with Whisper...")
    wav_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.wav')]
    wav_files = sorted(wav_files, key=lambda x: int(os.path.splitext(x)[0]))

    with open(output_csv, "w", encoding="utf-8") as transcript_file:
        for wav_file in wav_files:
            full_path = os.path.join(input_folder, wav_file)
            print(f"📝 Transcribing: {full_path}")

            result = model.transcribe(
                full_path,
                verbose=True,
                fp16=True,
                condition_on_previous_text=False,
                temperature=0.0,
                no_speech_threshold=0.1,
                logprob_threshold=-1.0,
                compression_ratio_threshold=2.4,
                language="en"
            )

            transcribed_text = result['text'].strip()
            transcript_file.write(f"wavs/{wav_file}|{transcribed_text}\n")

    print(f"✅ Transcription complete. See '{output_csv}'")

def main():
    base_dir = os.getcwd()
    original_audio_dir = base_dir  # Input files live in current dir
    split_dir = os.path.join(base_dir, "split_wav")
    converted_dir = os.path.join(base_dir, "wavs")

    check_ffmpeg_available()

    # Step 1: Split audio into chunks
    split_success = split_audio(original_audio_dir, split_dir)
    if not split_success:
        print("⚠️ Some files failed during splitting. Proceeding to conversion...")

    # Step 2: Convert split chunks to uniform format
    convert_success = convert_audio(split_dir, converted_dir)
    if convert_success:
        try:
            shutil.rmtree(split_dir)
            print(f"🧹 Removed temporary folder: {split_dir}")
        except Exception as e:
            print(f"⚠️ Could not remove split_wav folder: {e}")
    else:
        print("⚠️ Some files failed to convert. 'split_wav/' was not removed.")

    # Step 3: Transcribe with Whisper
    transcribe_audio(converted_dir, os.path.join(base_dir, "metadata.csv"))

if __name__ == "__main__":
    main()
