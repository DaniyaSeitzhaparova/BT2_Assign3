# Kazakhstan Constitutional AI Assistant

An intelligent legal research assistant that provides instant answers about Kazakhstan's Constitution using advanced AI technology. The system analyzes constitutional text and delivers accurate responses with direct citations to relevant articles.

## Features

- Document processing for PDF and DOCX files
- Chat interface for constitutional questions
- Source citation for answers
- Conversation history logging
- Ollama LLM integration

## Installation

1. First install Ollama from [ollama.ai](https://ollama.ai/)
2. Clone this repository:
```bash
git clone https://github.com/aiswlz/kazakhstan-ai-assistant.git
cd kazakhstan-ai-assistant
```
3. Install Python dependencies:

```bash
pip install -r requirements.txt
```
4. Download required models:
```bash
ollama pull llama2
ollama pull nomic-embed-text
```
## Usage
1. Run the application:
```bash
streamlit run app2.py
```
2. In the web interface:

- Configure Ollama settings in sidebar
- Upload Constitution documents (PDF/DOCX)
- Click "Process Documents"
- Ask questions in the chat interface
  
## Example Questions
Try asking:

- "What are the fundamental rights under Article 14?"
- "Explain the presidential election process"
- "What does the Constitution say about education?"
- "List the main principles in Chapter 1"
- "What are the amendment procedures?"

## Screenshots
1. Document Upload
![image](https://github.com/user-attachments/assets/560c2906-4537-48e4-a61a-df60dd9e4e65)
2. Chat Interface
![image](https://github.com/user-attachments/assets/2919396b-6bb9-4517-9f6f-2e88645f59a4)
3. Sources display
![image](https://github.com/user-attachments/assets/d6164bb1-faee-41ac-942f-d5bdb3f10c5a)

# LICENSE
MIT License

Copyright (c) [2025] [Astana IT University]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
