import logging
from pathlib import Path

import click

from video_similarity_search.backend.database_handler import MilvusDatabase, VideoToDatabase
from video_similarity_search.backend.embeddingextractor import VideoEmbeddingExtractor
from video_similarity_search.backend.model import CLIPModelProcessor
from video_similarity_search.backend.search import Search

logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


@click.command("video_similarity_search")
@click.option(
    "--video-folder",
    type=Path,
    required=True,
    help="Path to the folder containing videos.",
)
@click.option("--query", type=str, default="Person", help="Text query for searching videos.")
@click.option("--remove_old_data", is_flag=True, help="Text query for searching videos.")
def video_similarity_search(video_folder: Path, query: str, remove_old_data: bool):
    if not video_folder.exists():
        logger.error(f"Folder {video_folder} does not exist.")
        raise ValueError(f"Folder {video_folder} does not exist.")

    # Only CLIP model is supported right now
    # Initializing the model, video handler, milvus handler, video database and search
    model = CLIPModelProcessor()
    video_handler = VideoEmbeddingExtractor(model)
    milvus_handler = MilvusDatabase(remove_old_data=remove_old_data)
    video_database = VideoToDatabase(video_handler, milvus_handler)
    video_search = Search(model, milvus_handler)

    video_database.add_files_from_folder(video_folder)
    results = video_search.search_by_text(query)
    video_handler.present_query_results(results)
    logger.info(f"Search results: {results}")


cli.add_command(video_similarity_search)


if __name__ == "__main__":
    cli()
