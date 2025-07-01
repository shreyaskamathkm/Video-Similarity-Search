# Video Similarity Search

This repository provides a Python-based implementation for a Video Similarity Search System using Open-CLIP, OpenCV, and Milvus. The system allows you to index video frame embeddings into a Milvus database and perform similarity searches using text queries.

## Table of Contents

- [Video Similarity Search](#video-similarity-search)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Development Status](#development-status)
    - [Completed](#completed)
    - [Future Work (TODO)](#future-work-todo)
  - [File Structure](#file-structure)
  - [Acknowledgements](#acknowledgements)

## Features

- **Video Embedding Extraction**: Uses `open_clip_torch` to extract embeddings from video frames.
- **Efficient Frame Sampling**: Allows skipping frames to optimize processing.
- **Vector Database Integration**: Leverages Milvus for efficient storage and similarity search of embeddings.
- **Text-based Search**: Perform similarity search using text queries.
- **Video Management**: Add videos to the database, search, and query.

## Requirements

- Python 3.10+
- [Milvus](https://milvus.io/)
- [Poetry](https://python-poetry.org/)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd video_similarity_search
    ```

2.  **Install dependencies using Poetry:**
    ```bash
    poetry install
    ```

3.  **Set up Milvus:** Ensure Milvus is running. If not, run the following:
      ```bash
      bash standalone_embed.sh start
      ```

## Usage

The application is run via a command-line interface (CLI). The main command is `run_video_similarity`, which requires a configuration file.

1.  **Create a configuration file.** A sample configuration is provided in `config/video_search_similarity.yaml`. You can modify this file or create your own.

    ```yaml
    collection_name: "video_search"
    video_folder: "videos"
    query: "a man talking"
    frame_skip: 100
    reset_dataset: True
    model_name: "ViT-B-32-SigLIP2-256"
    model_architecture: "ViT-B-32-SigLIP2"
    model_pretrained: "webli"
    ```
    **Note:** It is recommended to set `reset_dataset: True` to ensure that the Milvus database is cleared before new videos are added. This prevents discrepancies and ensures a clean state for each run.

2.  **Run the application:**
    ```bash
    poetry run python -m video_similarity_search.cli run_video_similarity --config-path config/video_search_similarity.yaml
    ```

    This will:
    - Connect to the Milvus database.
    - Process the videos in the specified `video_folder`.
    - Embed the video frames using the specified CLIP model.
    - Store the embeddings in the Milvus collection.
    - Perform a similarity search with the provided `query`.
    - Display the search results.

## Development Status

### Completed

- [x] Text-based similarity search
- [x] Video processing and frame extraction
- [x] Milvus database integration
- [x] CLI for running the application

### Future Work (TODO)

- [ ] Implement image-based similarity search
- [ ] Add comprehensive unit and integration tests
- [ ] Add videos in existing Milvus database

## File Structure

```
.
├── config
│   └── video_search_similarity.yaml
├── video_similarity_search
│   ├── backend
│   │   ├── database_handler.py
│   │   ├── embeddingextractor.py
│   │   ├── model.py
│   │   ├── query_result_formatter.py
│   │   ├── search.py
│   │   ├── video_handler.py
│   │   ├── video_processor.py
│   │   └── video_segment_extractor.py
│   ├── cli.py
│   └── schema.py
├── videos
└── ...
```

- **`config/`**: Contains configuration files for the application.
- **`video_similarity_search/`**: Main application source code.
  - **`backend/`**: Core logic for video processing, database handling, and search.
  - **`cli.py`**: Command-line interface for the application.
  - **`schema.py`**: Pydantic models for application configuration.
- **`videos/`**: Default directory for storing input videos.

## Acknowledgements

- [OpenAI CLIP](https://github.com/openai/CLIP)
- [Milvus](https://milvus.io/)
- [OpenCV](https://opencv.org/)
- [open_clip_torch](https://github.com/mlfoundations/open_clip)
