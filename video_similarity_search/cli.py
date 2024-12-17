from pathlib import Path

from video_similarity_search.backend.database_handler import MilvusHandler, VideoDatabase
from video_similarity_search.backend.model import Model
from video_similarity_search.backend.search import VideoSearch
from video_similarity_search.backend.video_handler import VideoHandler


def main():
    model = Model()
    video_handler = VideoHandler(model)
    milvus_handler = MilvusHandler()
    video_database = VideoDatabase(model, video_handler, milvus_handler)

    # Example usage
    video_folder = Path("./videos")
    video_database.add_videos_from_folder(video_folder)

    milvus_handler.query(expr="id >= 0")

    video_search = VideoSearch(model, milvus_handler)
    query = "A person"
    results = video_search.search_by_text(query)
    video_handler.present_query_results(results)
    print("Search results:", results)


if __name__ == "__main__":
    main()
