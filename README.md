# MammoChat

MammoChat is an AI-powered medical information assistant designed to provide reliable breast cancer information from trusted sources. It uses RAG (Retrieval-Augmented Generation) to ensure accurate information delivery from reputable sources like BreastCancer.org and Komen.org.

## DEMO

Try the live demo at: [https://mammochat.streamlit.app](https://mammochat.streamlit.app)

## Features

- 🎯 **Source-Verified Information**: Only uses trusted medical sources like BreastCancer.org and Komen.org
- 🔍 **Smart Retrieval**: Uses embeddings and semantic search to find relevant information
- 💬 **Interactive Chat Interface**: Built with Streamlit for easy interaction
- 📚 **Document Processing**: Two-phase document processing pipeline for efficient content management
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

3. Configure secrets:

   The application supports two methods of configuration:

   **Option A: Streamlit Secrets (Recommended for Streamlit apps)**
   1. Copy the example secrets file:
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```
   2. Edit `.streamlit/secrets.toml` with your API keys:
   ```toml
   [api_keys]
   openai = "your-openai-api-key"
   supabase_url = "your-supabase-url"
   supabase_service_key = "your-supabase-service-key"
   ```
   Note: The `secrets.toml` file is excluded from version control for security.

   For Streamlit Cloud deployment:
   - Add these secrets via the Streamlit Cloud dashboard

   **Option B: Environment Variables (For scripts and non-Streamlit usage)**
   Create a `.env` file:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   SUPABASE_URL=your_supabase_url
   SUPABASE_SERVICE_KEY=your_supabase_service_key
   ```

   The application will first try to use Streamlit secrets, and fall back to environment variables if needed.

## Usage

### Chat Interface
Start the Streamlit app:
```bash
streamlit run app.py
```

Once the application is running, you can interact with the AI assistant through the web interface. Simply type your questions related to breast cancer, and the assistant will provide reliable information from trusted sources.

### Document Processing Pipeline

The project includes a comprehensive document processing pipeline (`process_documents.py`) that operates in two phases:

1. **Crawling & Storage Phase**
   ```bash
   # Crawl and store content without AI processing
   python process_documents.py crawl --source komen_org --urls komen.org --max-concurrent 5
   ```
   - Crawls websites and stores raw content
   - No OpenAI API calls during this phase
   - Configurable concurrency for efficient crawling
   - Automatic content chunking and storage of documents

2. **AI Processing Phase**
   ```bash
   # Process stored content with AI
   python process_documents.py process --batch-size 50 --max-retries 3
   ```
   - Processes stored content using OpenAI APIs
   - Generates embeddings, titles and summaries of chunks
   - Batch processing with progress tracking
   - Automatic retry mechanism for API calls

This two-phase approach offers several benefits:
- Separates crawling from AI processing for better cost management
- Allows for efficient re-processing of content if needed
- Provides robust error handling and logging
- Supports different content sources and processing configurations

## Project Structure

- `app.py`: Main Streamlit application with chat interface
- `MammoChat_agent.py`: Core agent logic and RAG implementation
- `process_documents.py`: Two-phase document processing pipeline
- `requirements.txt`: Project dependencies

## Architecture

### RAG System
The system uses a Retrieval-Augmented Generation (RAG) approach:
1. Documents are processed and stored with embeddings in Supabase
2. User queries are embedded and matched against the document database
3. Relevant content is retrieved and used to generate accurate responses

### Document Processing Pipeline
The pipeline is designed for efficiency and reliability:
- **Phase 1: Crawling & Storage**
  - Asynchronous web crawling with concurrency control
  - Content chunking with natural boundary detection
  - Raw storage without AI processing
  - Comprehensive error handling and logging

- **Phase 2: AI Processing**
  - Batch processing of stored content
  - Automatic embedding generation
  - Title and summary extraction
  - Progress tracking and retry mechanisms
  - Efficient database operations

### Database Schema
The pipeline uses two main tables in Supabase:
- `raw_pages`: Stores unprocessed content from crawling phase
- `processed_chunks`: Stores AI-processed chunks with embeddings

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [OpenAI](https://openai.com/) for providing the language model API
- [Supabase](https://supabase.com/) for the database infrastructure
- [Streamlit](https://streamlit.io/) for the web interface framework
- [BreastCancer.org](https://www.breastcancer.org) and [Komen.org](https://www.komen.org) for the medical information
