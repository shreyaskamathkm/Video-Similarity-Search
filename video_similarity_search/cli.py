import logging
from pathlib import Path

from video_similarity_search.backend.database_handler import MilvusDatabase, VideoToDatabase
from video_similarity_search.backend.embeddingextractor import VideoEmbeddingExtractor
from video_similarity_search.backend.model import CLIPModelProcessor
from video_similarity_search.backend.search import Search

logger = logging.getLogger(__name__)


def main():
    model = CLIPModelProcessor()
    video_handler = VideoEmbeddingExtractor(model)
    milvus_handler = MilvusDatabase()
    video_database = VideoToDatabase(video_handler, milvus_handler)

    # Example usage
    video_folder = Path("./videos")
    video_database.add_files_from_folder(video_folder)

    # milvus_handler.query(expr="id >= 0")

    video_search = Search(model, milvus_handler)
    query = "A person"
    results = video_search.search_by_text(query)
    video_handler.present_query_results(results)
    logger.info(f"Search results: {results}")


if __name__ == "__main__":
    main()
