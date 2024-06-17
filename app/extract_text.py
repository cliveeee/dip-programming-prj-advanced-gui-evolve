import os
import cv2
import pytesseract
import logging
from typing import Union, List, Dict
import utils
from utils import config
import json
import openai  # Importing openai module for openai_format_raw_ocr

# Setup logging
logging.basicConfig(level=logging.INFO)

class ExtractText:
    """
    A utility class for extracting and formatting code snippets from video frames using OCR and OpenAI.
    """

    @staticmethod
    def extract_code_in_range(filename: str, start_timestamp: float, end_timestamp: float) -> List[Dict[str, Union[str, float]]]:
        """
        Extract formatted code from a video file at 5-second intervals between start_time and end_time.
        :param filename: File path of the video to extract the frames from
        :param start_timestamp: Start time in seconds
        :param end_timestamp: End time in seconds
        :return: List of dictionaries containing the timestamp and the extracted/formatted code
        """
        extracted_code_data = []
        previous_text = ""

        for timestamp in range(int(start_timestamp), int(end_timestamp) + 1, 5):
            frame = ExtractText.extract_frame_at_timestamp(filename, timestamp)
            if frame is not None:
                extracted_text = pytesseract.image_to_string(frame)
                logging.info(f"Successfully extracted text from frame @ {timestamp}s in file {filename}")

                if not ExtractText.is_similar_text(previous_text, extracted_text):
                    if ExtractText.contains_code_keywords(extracted_text):
                        formatted_code = ExtractText.format_raw_ocr_string(extracted_text)
                        extracted_code_data.append({
                            'timestamp': timestamp,
                            'text': formatted_code
                        })
                        print(formatted_code)
                    previous_text = extracted_text
                else:
                    logging.info(f"Text at {timestamp}s is similar to the previous frame, ignoring.")
            else:
                logging.error(f"Unable to extract text from frame @ {timestamp}s in file {filename}")

        return extracted_code_data

    @staticmethod
    def is_similar_text(text1: str, text2: str) -> bool:
        """
        Check if two texts are similar.
        :param text1: The first text to compare
        :param text2: The second text to compare
        :return: True if texts are similar, otherwise False
        """
        return text1.strip() == text2.strip()

    @staticmethod
    def contains_code_keywords(text: str) -> bool:
        """
        Check if the extracted text contains specific code-related keywords.
        :param text: The extracted text to check
        :return: True if keywords are found, otherwise False
        """
        keywords = ['def', 'class', 'import', 'print', 'for', 'while', 'if', 'else', 'try', 'except', 'public']
        return any(keyword in text for keyword in keywords)

    @staticmethod
    def format_raw_ocr_string(extracted_text: str) -> str:
        """
        Attempts to format a given string to match given programming language
        :param extracted_text: Raw OCR text to format
        :return: Formatted text as string
        """
        language = config("UserSettings", "programming_language")
        formatted_text = ExtractText.openai_format_raw_ocr(extracted_text, language)
        return formatted_text

    @staticmethod
    def extract_frame_at_timestamp(filename: str, timestamp: float) -> Union[None, bytes]:
        """
        Extract a frame from a video at a given timestamp
        :param filename: File path to extract frame from
        :param timestamp: Timestamp to extract the frame from
        :return: Returns capture frame or None
        """
        video_file_path = os.path.join('out', 'videos', filename)
        cap = cv2.VideoCapture(video_file_path)
        if not cap.isOpened():
            logging.error(f"Failed to open {video_file_path} stream")
            return None
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()
        if ret:
            cap.release()
            logging.info(f"Successfully captured frame @ {timestamp}s in file {video_file_path}")
            return frame
        else:
            logging.error(f"Failed to capture frame @ {timestamp}s in file {video_file_path}")
            return None

    @staticmethod
    def openai_format_raw_ocr(extracted_text: str, language: str) -> str:
        """
        Format given text using OpenAI language model API
        :param extracted_text: Raw extracted text to format
        :param language: Programming language to format the raw text as
        :return: Formatted code as string
        """
        try:
            prompt = f"Fix up the following {language} code snippet, fix up any indentation errors, syntax errors, " \
                     f"and anything else that is incorrect: '{extracted_text}'"
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system",
                     "content": f"You are a coding assistant. You reply only in {language} code "
                                "that is correct and formatted. Do NOT reply with any explanation, "
                                f"only code. If you are given something that is not {language} code, "
                                "you must NOT include it in your response. If nothing is present, "
                                "simply return 'ERROR' and nothing else. Do NOT return leading or "
                                "trailing backticks and do NOT return the language before the code snippet."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message['content']
        except openai.OpenAIError as error:
            logging.exception(error)
            return extracted_text

    @staticmethod
    def save_to_json(data: List[Dict[str, Union[str, float]]], output_file_path: str):
        """
        Save the extracted code data to a JSON file.
        :param data: The list of dictionaries containing the extracted code data
        :param output_file_path: The file path where the JSON data will be saved
        """
        try:
            with open(output_file_path, 'w') as json_file:
                json.dump(data, json_file, indent=4)
            logging.info(f"Extracted code data has been saved to {output_file_path}")
        except Exception as e:
            logging.error(f"Failed to save extracted code data to {output_file_path}: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Usage: python script_name.py <video_filename> <start_time> <end_time>")
        sys.exit(1)

    video_filename = sys.argv[1]
    start_time = float(sys.argv[2])
    end_time = float(sys.argv[3])

    extractor = ExtractText()
    extracted_code_data = extractor.extract_code_in_range(video_filename, start_time, end_time)

    output_file_path = "extracted_code_data.json"
    ExtractText.save_to_json(extracted_code_data, output_file_path)
