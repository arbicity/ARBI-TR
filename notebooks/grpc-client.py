import grpc
import audio_processing_pb2
import audio_processing_pb2_grpc
import json  # Add this import

def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = audio_processing_pb2_grpc.AudioProcessingServiceStub(channel)

    # Define settings for the audio processing
    settings = audio_processing_pb2.AudioSettings(
        size_of_model='large',
        task='transcribe',
        source_language='en',
        speaker_number=2
    )

    # Function to generate the request stream including settings and audio content
    def request_stream():
        # First send the settings
        yield audio_processing_pb2.AudioRequest(settings=settings)

        # Then stream the audio file as chunks
        with open("svamc.mp3", "rb") as f:
            while (chunk := f.read(1024)):
                yield audio_processing_pb2.AudioRequest(chunk=audio_processing_pb2.AudioChunk(content=chunk))

    # Send the audio processing request to the server
    response = stub.ProcessAudio(request_stream())

    # Parse the JSON string received from the server and output it
    grouped_segments = json.loads(response.message)
    print("Grouped segments:", grouped_segments)

if __name__ == '__main__':
    run()
