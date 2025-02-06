# MammoChat

MammoChat is an AI-powered medical information assistant designed to provide reliable breast cancer information from trusted sources. It uses RAG (Retrieval-Augmented Generation) to ensure accurate information delivery from reputable sources like BreastCancer.org and Komen.org.

## Features

- 🎯 **Source-Verified Information**: Only uses trusted medical sources like BreastCancer.org and Komen.org
- 🔍 **Smart Retrieval**: Uses embeddings and semantic search to find relevant information
- 💬 **Interactive Chat Interface**: Built with Streamlit for easy interaction
- 📚 **Document Processing**: Asynchronous processing of medical documents with automatic embedding generation
- 🔄 **Real-time Updates**: Streaming responses for better user experience

## Prerequisites

- Python 3.8+
- OpenAI API key
- Supabase account and credentials

## Installation

1. Clone the repository:
```bash
git clone https://github.com/HadleyLab/MammoChat.git
cd MammoChat
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env`:
```env
OPENAI_API_KEY=your_openai_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_supabase_service_key
```

## Usage

Start the Streamlit app:
```bash
streamlit run MammoChat.py
```

Once the application is running, you can interact with the AI assistant through the web interface. Simply type your questions related to breast cancer, and the assistant will provide reliable information from trusted sources.

## Project Structure

- `MammoChat.py`: Main Streamlit application with chat interface
- `MammoChat_agent.py`: Core agent logic and RAG implementation
- `async_document_processor.py`: Asynchronous document processing and embedding generation
- `requirements.txt`: Project dependencies

## Architecture

### RAG System
The system uses a Retrieval-Augmented Generation (RAG) approach:
1. Documents are processed and stored with embeddings in Supabase
2. User queries are embedded and matched against the document database
3. Relevant content is retrieved and used to generate accurate responses

### Document Processing
- Asynchronous processing with configurable batch sizes
- Automatic embedding generation using OpenAI
- Title and summary generation for better organization
- Health checks and error handling for production reliability

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [OpenAI](https://openai.com/) for providing the language model API
- [Supabase](https://supabase.com/) for the database infrastructure
- [Streamlit](https://streamlit.io/) for the web interface framework
- [BreastCancer.org](https://www.breastcancer.org) and [Komen.org](https://www.komen.org) for the medical information
