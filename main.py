import os
import wave_analise
import wave_processor
from datetime import datetime
def main():
        directory = './input'
        if os.path.exists(directory):
        # Iterate over all files in the directory
                for archive in os.listdir(directory):
                        filepath = os.path.join(directory, archive)
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        buffer = wave_analise.save_audio_in_chunks(filepath, chunk_size_ms=100)
                        processed_waves = wave_processor.process(buffer)
                        
                        # Define filename with timestamp
                        filename = f"output_{timestamp}_{archive}.txt"
                        print(processed_waves)
                        # Save output to file
                        with open('./output/' + filename, 'w') as f:
                                f.write(str(processed_waves))
        return "OK"
main()