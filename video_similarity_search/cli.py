from pathlib import Path

import click

from video_similarity_search.backend.database_handler import MilvusHandler, VideoDatabase
from video_similarity_search.backend.model import Model
from video_similarity_search.backend.search import VideoSearch
from video_similarity_search.backend.video_handler import VideoHandler


@click.command()
@click.option(
    "--video-folder",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default="./videos",
    help="Path to the folder containing videos to process.",
)
@click.option(
    "--query",
    type=str,
    default="A person",
    help="Text query for video similarity search.",
)
def main(video_folder: Path, query: str) -> None:
    model = Model()
    video_handler = VideoHandler(model)
    milvus_handler = MilvusHandler()
    video_database = VideoDatabase(model, video_handler, milvus_handler)

    # Example usage
    video_database.add_videos_from_folder(video_folder)

    milvus_handler.query(expr="id >= 0")

    video_search = VideoSearch(model, milvus_handler)
    results = video_search.search_by_text(query)
    video_handler.present_query_results(results)
    print("Search results:", results)


if __name__ == "__main__":
    main()
