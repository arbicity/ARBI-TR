import grpc
from concurrent import futures
import os
import tempfile
import logging
import audio_processing_pb2
import audio_processing_pb2_grpc
import utils  # Assuming the utils.py contains your audio processing logic
import json  # Add this import

class AudioProcessingService(audio_processing_pb2_grpc.AudioProcessingServiceServicer):

    def ProcessAudio(self, request_iterator, context):
        settings = None
        temp_audio_file_path = None

        try:
            for request in request_iterator:
                if request.HasField('settings'):
                    settings = request.settings
                    logging.info(f"Received settings: model={settings.size_of_model}, task={settings.task}, source_language={settings.source_language}, speaker_number={settings.speaker_number}")
                elif request.HasField('chunk'):
                    if not temp_audio_file_path:
                        # Create a temporary file to store incoming audio chunks
                        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                        temp_audio_file_path = temp_audio_file.name
                    # Write chunks to the temporary file
                    temp_audio_file.write(request.chunk.content)
            
            if temp_audio_file_path:
                # Close the file after writing all chunks
                temp_audio_file.close()

            if settings and temp_audio_file_path:
                # Process the audio file using utils
                results = utils.process_audio(
                    file_path=temp_audio_file_path,
                    size_of_model=settings.size_of_model,
                    task=settings.task,
                    source_language=settings.source_language,
                    speaker_number=settings.speaker_number
                )
                logging.info("Audio processing completed successfully.")
                
                # Return the grouped segments as a JSON string
                grouped_segments_json = json.dumps(results)
                return audio_processing_pb2.ProcessResponse(message=grouped_segments_json)
            else:
                logging.error("Failed to process audio due to missing settings or audio data.")
                return audio_processing_pb2.ProcessResponse(message="Failed to process audio due to missing settings or audio data.")

        except Exception as e:
            logging.error(f"Error processing audio: {e}")
            return audio_processing_pb2.ProcessResponse(message=f"Error during audio processing: {str(e)}")
        finally:
            if temp_audio_file_path and os.path.exists(temp_audio_file_path):
                os.remove(temp_audio_file_path)
                logging.info(f"Temporary file {temp_audio_file_path} removed.")

    def GetResults(self, request, context):
        # Implement this based on how results should be retrieved, e.g., from a database or file system
        pass

def serve():
    logging.basicConfig(level=logging.INFO)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    audio_processing_pb2_grpc.add_AudioProcessingServiceServicer_to_server(
        AudioProcessingService(), server)
    port = '50051'
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    logging.info(f"Server started and listening on port {port}.")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info("Server shutdown requested.")
        server.stop(0)

if __name__ == '__main__':
    serve()
