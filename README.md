# Video Similarity Search System

This repository provides a Python-based implementation for a Video Similarity Search System using CLIP, OpenCV, and Milvus. The system allows you to index video frame embeddings into a Milvus database and perform similarity searches using text or image queries.

______________________________________________________________________

## Table of Contents

- [Video Similarity Search System](#video-similarity-search-system)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Requirements](#requirements)
    - [Python Dependencies](#python-dependencies)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Acknowledgements](#acknowledgements)

______________________________________________________________________

## Features

- **Video Embedding Extraction**: Uses OpenAI's CLIP model to extract embeddings from video frames.
- **Efficient Frame Sampling**: Allows skipping frames to optimize processing.
- **Vector Database Integration**: Leverages Milvus for efficient storage and similarity search of embeddings.
- **Search Modes**:
  - Text-based search
  - Image-based search
- **Video Management**: Add videos to the database, search, query, and delete videos from the collection.

______________________________________________________________________

## Requirements

- Python 3.8+
- [Milvus 2.0](https://milvus.io/)

### Python Dependencies

- `opencv-python`
- `numpy`
- `transformers`
- `pillow`
- `pymilvus`

______________________________________________________________________

## Installation

1. Clone the repository:

   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

1. Install dependencies using Poetry:

   ```bash
   poetry install
   ```

1. Set up Milvus:

   - Ensure Milvus is running locally or remotely.
   - Update the connection URI in the `MilvusHandler` class (default is `http://localhost:19530`).
   - If Milvus client is already installed, you can start it by using

   ```bash
   bash standalone_embed.sh
   ```

______________________________________________________________________

## Usage

You can run the code by using the following CLI:

```python
vss --video-folder <path-to-the-video-folder> --remove_old_data
```

```python
vss --help
```

______________________________________________________________________

## Acknowledgements

- [OpenAI CLIP](https://github.com/openai/CLIP)
- [Milvus](https://milvus.io/)
- [OpenCV](https://opencv.org/)
