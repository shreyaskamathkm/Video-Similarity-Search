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
    - [Adding Videos to the Database](#adding-videos-to-the-database)
    - [Text-based Search](#text-based-search)
    - [Image-based Search](#image-based-search)
    - [Managing the Database](#managing-the-database)
  - [File Structure](#file-structure)
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

______________________________________________________________________

## Usage

### Adding Videos to the Database

To add videos from a folder to the database:

```bash
python main.py
```

Modify the `video_folder` variable in the `__main__` function to specify the path to your video directory.

### Text-based Search

Perform a search using a text query:

```python
query = "A person stealing a bag"
results = video_search.search_by_text(query)
print("Search results:", results)
```

### Image-based Search

Perform a search using an image query:

```python
from PIL import Image
image = Image.open("query_image.jpg")
results = video_search.search_by_image(image)
print("Search results:", results)
```

### Managing the Database

- Query videos:
  ```python
  milvus_handler.query(expr="id >= 0")
  ```
- Delete a specific video:
  ```python
  milvus_handler.delete_video("video_name.mp4")
  ```
- Drop the entire collection:
  ```python
  milvus_handler.delete_collection()
  ```

______________________________________________________________________

## File Structure

- `main.py`: Main script to add videos and perform searches.
- `handlers/`:
  - `model.py`: CLIP model operations.
  - `video_handler.py`: Handles video processing and frame embedding extraction.
  - `milvus_handler.py`: Handles Milvus operations including saving and searching embeddings.
  - `search.py`: Search operations for text and image queries.

______________________________________________________________________

## Acknowledgements

- [OpenAI CLIP](https://github.com/openai/CLIP)
- [Milvus](https://milvus.io/)
- [OpenCV](https://opencv.org/)