import json
from datetime import datetime
from typing import List, Dict
import logging
from extract_text import ExtractText  # Assuming ExtractText is correctly imported from the extract_text module
import sys

# Setup logging
logging.basicConfig(level=logging.INFO)

class BookmarkManager:
    @staticmethod
    def generate_bookmarks_from_json(input_file_path: str) -> List[Dict[str, str]]:
        """
        Generate bookmarks from JSON data.
        :param input_file_path: Path to the input JSON file
        :return: List of dictionaries representing bookmarks
        """
        try:
            with open(input_file_path, 'r') as json_file:
                data = json.load(json_file)
            logging.info(f"Data successfully loaded from {input_file_path}")
        except Exception as e:
            logging.error(f"Failed to load data from {input_file_path}: {e}")
            return []

        bookmarks = []
        for entry in data:
            timestamp = entry.get("timestamp")
            code = entry.get("text")
            if not timestamp or not code:
                logging.warning(f"Invalid entry found and skipped: {entry}")
                continue


            bookmark = {timestamp: code}

            bookmarks.append(bookmark)

        return bookmarks

    @staticmethod
    def save_bookmarks_to_json(bookmarks: List[Dict[str, str]], output_file_path: str):
        """
        Save bookmarks to a JSON file.
        :param bookmarks: List of dictionaries representing bookmarks
        :param output_file_path: Path to the output JSON file
        """
        try:
            with open(output_file_path, 'w') as json_file:
                json.dump(bookmarks, json_file, indent=4)
            logging.info(f"Bookmarks have been saved to {output_file_path}")
        except Exception as e:
            logging.error(f"Failed to save bookmarks to {output_file_path}: {e}")

# Example usage for testing
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python bookmark_manager.py <input_json_file> <start_time> <end_time> <video_filename>")
        sys.exit(1)

    input_file_path = sys.argv[1]
    start_time = float(sys.argv[2])
    end_time = float(sys.argv[3])
    video_filename = sys.argv[4]

    extractor = ExtractText()
    extracted_code_data = extractor.extract_code_in_range(video_filename, start_time, end_time)


    bookmarks = []
    for entry in extracted_code_data:
        timestamp = entry['timestamp']
        code = entry['text']
        time_str = datetime.fromisoformat(str(timestamp)).strftime('%H:%M')
        bookmark = {time_str: code}
        bookmarks.append(bookmark)

    # Output the bookmarks (for testing purposes)
    print("Generated Bookmarks:")
    for bookmark in bookmarks:
        print(bookmark)

    output_bookmarks_path = 'bookmarks.json'
    BookmarkManager.save_bookmarks_to_json(bookmarks, output_bookmarks_path)
