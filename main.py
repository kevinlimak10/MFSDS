import os
import wave_analise
import wave_processor
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import PhotoImage

root = tk.Tk()

def read_wav_file():
    # Open file dialog and get the path of the selected file
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if not file_path:
        print("No file selected.")
        return

    return file_path

def obterFormula(nome_arquivo, identificador):
    with open(nome_arquivo, 'r') as arquivo:
        for linha in arquivo:
            if linha.startswith(identificador + "="):
                return linha.split("=", 1)[1].strip()
    return None

def main():
        filepath = read_wav_file()
        if not filepath:
                return 
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        root.title("Data Table")
        root.config(padx=10, pady=100)
        #apply dft
        buffer = wave_analise.save_audio_in_chunks(filepath, chunk_size_ms=100)
        processed_waves = wave_processor.process(buffer)
        
        # Define filename with timestamp
        filename = f"output_{timestamp}_processed.txt"
        print(processed_waves)
        # Save output to file
        with open('./output/' + filename, 'w') as f:
                f.write(str(processed_waves))
        # The data to be displayed

                # Create the treeview
        tree = ttk.Treeview(root, columns=("analised", "formula", "percent"), show="headings")
        tree.heading("analised", text="Analised")
        tree.heading("formula", text="Formula")
        tree.heading("percent", text="Percent")

        # Add data to the treeview
        for item in processed_waves:
            nome_sem_extensao, _ = os.path.splitext(item["simility"]["formula"])
            formula = obterFormula(nome_arquivo="./generateFormula/function.txt", identificador=nome_sem_extensao)
            tree.insert("", tk.END, values=(item["analised"], formula, item["simility"]["percent"]))

        # Pack the treeview into the window
        tree.pack(expand=True, fill=tk.BOTH)
        # Start the GUI event loop
        root.mainloop()

if __name__ == '__main__':
    main()


