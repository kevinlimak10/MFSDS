import logging
import wave
import numpy as np
import matplotlib.pyplot as plt
import io


def save_audio_in_chunks(input_data, chunk_size_ms=100):
    figures_data = []
    try:
        with wave.open(input_data, 'rb') as wave_file:
            # Get the parameters of the wave_analise.py file
            num_channels = wave_file.getnchannels()
            sample_width = wave_file.getsampwidth()
            frame_rate = wave_file.getframerate()
            num_frames = wave_file.getnframes()

            # Calculate chunk size in frames
            chunk_size_frames = int(chunk_size_ms / 1000 * frame_rate)

            # Iterate over the frames in chunks
            for i in range(0, num_frames, chunk_size_frames):
                # Read chunk of frames
                frames = wave_file.readframes(chunk_size_frames)

                # Convert frames to numpy array
                frames_np = np.frombuffer(frames, dtype=np.int16)

                # If stereo, separate channels
                if num_channels == 2:
                    left_channel = frames_np[::2]
                    right_channel = frames_np[1::2]
                else:
                    left_channel = frames_np
                    right_channel = None

                # Calculate time axis for the chunk
                time = np.arange(i, min(i + chunk_size_frames, num_frames)) / frame_rate

                # Plot
                plt.figure(figsize=(10, 4))
                plt.plot(time, left_channel, label='Left Channel', color='b')
                if right_channel is not None:
                    plt.plot(time, right_channel, label='Right Channel', color='r')
                plt.xlabel('Time [s]')
                plt.ylabel('Amplitude')
                title = f'Chunk_{i // chunk_size_frames}'
                plt.title(title)
                plt.legend()
                plt.grid(True)

                # Save figure to in-memory buffer
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)

                # Append figure data along with binary data and title
                figures_data.append({
                    id: i,
                    'title': title,
                    'figure_buffer': buf
                })

                # Close figure to free memory
                plt.close()
    except Exception as e:
        logging.error(f'has error:{e}')
    return figures_data

