import azure.functions as func
import logging
import main

app = func.FunctionApp()

@app.blob_trigger(arg_name="myblob", path="mycontainer",
                               connection="9881fe_STORAGE") 
def blob_trigger(myblob: func.InputStream):
    logging.info(f"Python blob trigger function processed blob"
                f"Name: {myblob.name}"
                f"Blob Size: {myblob.length} bytes")
    binary_data = myblob.read()
    main.save_audio_in_chunks(binary_data,chunk_size_ms=100)
