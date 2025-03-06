import json
import time
import io
import base64
import boto3
import os
from pydub import AudioSegment
import subprocess
import re

# Initialize S3 client
s3_client = boto3.client('s3')


def lambda_handler(event, context):
    """Lambda function to trigger processing on file upload to S3"""

    # check_ffmpeg()
    # check_ffprobe()

    bucket, key = extract_event_details(event)

    print(f"bucket >>> {bucket}")
    print(f"KEY >>> {key}")

    directory_name = extract_first_subdirectory(key)

    print(f"directory_name >>> {directory_name}")

    serialized_audio_key = f"{directory_name}/serialized_audio_segments.json"

    print(f"serialized_audio_key {serialized_audio_key}")

    diarization_dict = json.loads(get_object_from_s3(bucket=bucket, key=key))

    print(f"diarization_dict {diarization_dict}")

    # Find the wav file in the specified directory and return its path
    wav_file_path = directory_name + "/audio.wav"

    print(f"wav_file_path {wav_file_path[:10]}")

    print(f"bucket >>> {bucket} wav_file_path >>> {wav_file_path}")

    # Retrieve the audio file from S3
    audio_bytes = get_object_from_s3(bucket=bucket, key=wav_file_path)

    process_audio_and_upload_serialized_audio_segments(audio_bytes, diarization_dict, bucket, serialized_audio_key)

    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "Audio processing complete"
        }),
    }


def process_audio_and_upload_serialized_audio_segments(audio_file, diarization_dict, bucket_name, s3_key):
    """
    Load a large audio file, split it into segments, encode each segment to base64,
    serialize the segments to JSON, and upload to S3.

    Args:
        audio_file_path (str): Path to the large audio file.
        bucket_name (str): Name of the S3 bucket.
        s3_key (str): Key path in the S3 bucket where the JSON will be stored.
        :param audio_file:
        :param diarization_dict:
    """

    # Split the audio into segments
    audio_buffer_list = split_audio_into_segments(audio_file, diarization_dict)

    # Encode each segment to base64 so we can serialize it
    encoded_audio_list = encode_audio_buffers(audio_buffer_list)

    # Serialize the list of base64-encoded audio segments to JSON
    serialized_audio_segments = json.dumps(encoded_audio_list)

    # Upload the JSON string to S3
    s3_client.put_object(
        Bucket=bucket_name,
        Key=s3_key,
        Body=serialized_audio_segments,
        ContentType='application/json'
    )

    print(f"Serialized audio segments saved to s3://{bucket_name}/{s3_key}")


def split_audio_into_segments(audio_file, diarization_dict):
    """
    Splits an audio file into smaller segments based on the specified segment length from the diarization dictionary.
    """
    # Load the audio file as an AudioSegment
    sound = AudioSegment.from_file(io.BytesIO(audio_file))

    # Initialize variables
    audio_buffer_list = []
    start_time_list = []
    end_time_list = []
    total_clipping_time = 0
    n_sentences = len(diarization_dict)

    # Iterate over diarization segments
    for i, segment_dict in diarization_dict.items():
        start_time = segment_dict["start"]
        end_time = segment_dict["end"]

        # Append start and end times to lists
        start_time_list.append(start_time)
        end_time_list.append(end_time)

        # Measure time to get audio segment
        st_get_clip = time.perf_counter()
        segment = get_clip_buffer(sound, start_time, end_time, target_audio_format="mp3")
        et_get_clip = time.perf_counter()
        get_clip_time = et_get_clip - st_get_clip

        # Update total clipping time and add audio buffer to the list
        total_clipping_time += get_clip_time
        audio_buffer_list.append(segment)

    return audio_buffer_list


def get_clip_buffer(original_sound_bytes: bytes,
                    start_time: int,
                    end_time: int,
                    source_audio_format: str = None,
                    target_audio_format: str = "mp3",
                    target_nbr_of_channels: int = 1) -> bytes:
    """
       Extracts a portion of audio, adjusts its channels, and returns the result as a bytes buffer.

       Parameters:
       - original_sound_bytes (bytes): The binary representation of the original audio.
       - start_time (float): The starting time for extracting audio in seconds.
       - end_time (float): The ending time for extracting audio in seconds.
       - source_audio_format (str, optional): The format of the source audio file. If None, assumes original_sound_bytes is in the desired format.
       - target_audio_format (str, optional): The desired format for the target audio. Default is "wav".
       - target_nbr_of_channels (int, optional): The desired number of channels for the target audio. Default is 1.

       Returns:
       - io.BytesIO: A bytes buffer containing the modified audio data in the specified format.

       Usage:
       This function is designed to extract a segment of audio, adjust its channels, and return the result as a bytes buffer.
       If source_audio_format is provided, it loads the audio from the bytes using the specified format; otherwise, it assumes original_sound_bytes is in the desired format.
       The extracted audio is trimmed based on start_time and end_time, and its number of channels is set to target_nbr_of_channels.
       The modified audio is then exported to a bytes buffer in the specified target_audio_format, and the buffer is returned.

       """
    if source_audio_format is not None:
        # AudioSegment is capable of reading a file directly from the bytes (using a stream)
        sound = AudioSegment.from_file(io.BytesIO(original_sound_bytes), source_audio_format)
    else:
        sound = original_sound_bytes

    sound = sound[start_time * 1000: end_time * 1000]  # times must be in milliseconds
    sound = sound.set_channels(target_nbr_of_channels)  # embeddings are calculated from 1D audio files

    sound_buffer = io.BytesIO()
    sound.export(sound_buffer, format=target_audio_format)

    return sound_buffer


def extract_first_subdirectory(key: str) -> str:
    match = re.match(r'([^/]+)/', key)
    if match:
        return match.group(1)
    raise ValueError("No subdirectory found in the key.")


def encode_audio_buffers(audio_buffers: list) -> list:
    """
    Encodes a list of io.BytesIO audio buffers to base64-encoded strings.

    Parameters:
    - audio_buffers (list): List of io.BytesIO objects containing audio data.

    Returns:
    - list: List of base64-encoded audio strings.
    """
    # Use Joblib with 'threading' backend to avoid disk usage
    # encoded_audio_list = Parallel(n_jobs=1, backend='threading')(
    #     delayed(encode_audio_segment)(AudioSegment.from_file(audio_buffer, format="mp3"))
    #     for audio_buffer in audio_buffers
    # )
    # return encoded_audio_list

    encoded_audio_list = []
    for audio_buffer in audio_buffers:
        audio_segment = AudioSegment.from_file(audio_buffer, format="mp3")  # Load audio segment
        encoded_audio = encode_audio_segment(audio_segment)  # Encode to base64
        encoded_audio_list.append(encoded_audio)
    return encoded_audio_list


def extract_event_details(event: dict) -> tuple:
    """
    Extract the bucket name and key from the S3 event.

    Args:
        event (dict): The event payload that provides metadata regarding the uploaded file in the S3 bucket to which the Lambda function is subscribed via an event trigger. This payload typically includes details such as the S3 bucket name, object key, and other relevant information necessary for processing the file.

    Returns:
        tuple: The bucket name and the key of the S3 object.
    """
    try:
        bucket = event["Records"][0]["s3"]["bucket"]["name"]
        key = event["Records"][0]["s3"]["object"]["key"]

        return bucket, key
    except KeyError as e:
        raise ValueError(f"Missing expected key in event: {e}")


def get_object_from_s3(bucket: str, key: str) -> bytes:
    """
    Get an object from S3 as bytes.

    Args:
        bucket (str): The name of the S3 bucket.
        key (str): The key of the S3 object.

    Returns:
        bytes: The content of the S3 object.
    """
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        return response['Body'].read()
    except ClientError as e:
        raise ValueError(f"Error fetching object from S3: {e}")


def encode_audio_segment(audio_segment: AudioSegment) -> str:
    """
    Encodes an AudioSegment object to a base64-encoded string.
    """
    buffer = io.BytesIO()
    audio_segment.export(buffer, format="mp3")  # Export as MP3 format
    encoded_audio = base64.b64encode(buffer.getvalue()).decode('utf-8')  # Encode and convert to string
    return encoded_audio
