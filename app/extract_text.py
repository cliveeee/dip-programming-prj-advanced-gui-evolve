import types

import openai
import cv2
import pytesseract
import logging
from typing import Union, List, Dict
import utils
from utils import config
import json


class ExtractText:
    """
    A utility class for extracting and formatting code snippets from video frames using OCR and OpenAI.

    This class provides methods for extracting code from a video frame, formatting the code to match a
    specific programming language, and utilizing the OpenAI language model for code correction.

    Note:
    This class assumes the availability of the OpenAI GPT-3.5 Turbo model and PyTesseract for OCR.
    """

    @staticmethod
    def extract_code_in_range(filename: str, start_timestamp: float, end_timestamp: float) -> List[Dict[str, Union[str, float]]]:
        """
        Extract formatted code from a video file at 5-second intervals between start_time and end_time.
        :param filename: File path of the video to extract the frames from
        :param start_time: Start time in seconds
        :param end_time: End time in seconds
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
    def extract_frame_at_timestamp(filename: str, timestamp: float) -> Union[cv2.VideoCapture, None]:
        """
        Extract a frame from a video at a given timestamp
        :param filename: File path to extract frame from
        :param timestamp: Timestamp to extract the frame from
        :return: Returns capture frame or None
        """
        cap = cv2.VideoCapture(f"{utils.get_vid_save_path()}{filename}")
        if not cap.isOpened():
            logging.error(f"Failed to open {filename} stream")
            return None
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()
        if ret:
            cap.release()
            logging.info(f"Successfully captured frame @ {timestamp}s in file {filename}")
            return frame
        else:
            logging.error(f"Failed to capture frame @ {timestamp}s in file {filename}")
            return None

    @staticmethod
    def openai_format_raw_ocr(extracted_text: str, language: str) -> str:
        """
        "Given an input of potentially raw OCR capture from a video containing Python code, your task is to correct
        and format the code. Ensure the code's indentation and syntax are accurate. Exclude any content that isn't
        valid Python code. If no recognizable Python content is detected, return 'ERROR'. Do not provide
        explanations, leading or trailing backticks, or specify the language in your response. Simply return the
        corrected code. Avoid making extensive alterations; the goal is to retain the original intent of the capture
        as closely as possible."

        prompt = f"Fix up the following {language} code snippet: '{extracted_text}'" response =
        openai.ChatCompletion.create( model="gpt-3.5-turbo", messages=[ {"role": "system", "content": "Given an input
        of potentially raw OCR capture from a video containing code, your task is to correct and format the code.
        Ensure the code's indentation and syntax are accurate. Exclude any content that isn't valid for the specified
        language. If no recognizable content is detected, return 'ERROR'. Do not provide explanations, leading or
        trailing backticks, or specify the language in your response. Simply return the corrected code. Avoid making
        extensive alterations; the goal is to retain the original intent of the capture as closely as possible."},
        {"role": "user", "content": prompt} ] ) return response.choices[0].message['content']

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
                                "trailing"
                                "backticks and do NOT return the language before the code snippet."},
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


# test
start_time = 201  # Start timestamp in seconds
end_time = 223    # End timestamp in seconds

extractor = ExtractText()
extracted_code_data = extractor.extract_code_in_range("oop(1).mp4", start_time, end_time)

output_file_path = "extracted_code_data.json"
ExtractText.save_to_json(extracted_code_data, output_file_path)

