import os
from PyPDF2 import PdfReader

class PDFToText:
    def convert_to_text(self,pdf_path: str):
        # Create a directory for transcripts if it doesn't exist
        transcripts_dir = 'transcripts'
        os.makedirs(transcripts_dir, exist_ok=True)

        # Generate the transcript file name based on the PDF file name
        file_name = os.path.splitext(os.path.basename(pdf_path))[0]
        transcript_file_name = os.path.join(transcripts_dir, file_name + '.txt')

        # Read the PDF file
        reader = PdfReader(pdf_path)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'

        # Save the extracted text to a file
        with open(transcript_file_name, 'w') as file:
            file.write(text)
        print(f'Transcript saved to {transcript_file_name}')

    def convert_directory_to_text(self, directory_path):
        # Create a directory for transcripts if it doesn't exist
        transcripts_dir = 'transcripts'
        os.makedirs(transcripts_dir, exist_ok=True)

        # Iterate over all files in the directory
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.pdf'):
                    pdf_path = os.path.join(root, file)
                    # Generate the transcript file name based on the PDF file name
                    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
                    transcript_file_name = os.path.join(transcripts_dir, file_name + '.txt')

                    # Read the PDF file
                    reader = PdfReader(pdf_path)
                    text = ''
                    for page in reader.pages:
                        text += page.extract_text() + '\n'

                    # Save the extracted text to a file
                    with open(transcript_file_name, 'w') as file:
                        file.write(text)
                    print(f'Transcript saved to {transcript_file_name}')

# Example usage
pdf_converter = PDFToText()
pdf_converter.convert_directory_to_text('/Users/vikasvashistha/github/personal-assistant/pdf-files')
