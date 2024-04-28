import wave_analise
import wave_processor

def main():
        archive = './entry/input.wav'
        buffer = wave_analise.save_audio_in_chunks(archive, chunk_size_ms=100)
        processed_waves = wave_processor.process(buffer)
        return "OK"

main()