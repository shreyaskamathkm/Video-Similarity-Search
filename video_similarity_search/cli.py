import logging
from pathlib import Path

import click
from cloudpathlib import AnyPath, S3Path

from video_similarity_search.backend.database_handler import MilvusHandler, VideoDatabase
from video_similarity_search.backend.model import ClipModel
from video_similarity_search.backend.search import VideoSearch
from video_similarity_search.backend.video_handler import VideoHandler

logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


@cli.command("run_video_similarity")
@click.option(
    "--video-folder",
    type=AnyPath,
    default="./videos",
    help="Path to the folder containing videos to process.",
)
@click.option(
    "--query",
    type=str,
    default="A person",
    help="Text query for video similarity search.",
)
@click.option(
    "--collection_name",
    type=str,
    default="video_search_similarity",
    help="Name of the collection.",
)
@click.option(
    "--reset-dataset",
    is_flag=True,
    default=True,
    help="Wether to drop reset the Milvus dataset.",
)
@click.option(
    "--frame-skip",
    type=int,
    default=2,
    help="Number of frames to skip in a video.",
)
def run_video_similarity(
    video_folder: S3Path | Path,
    query: str,
    collection_name: str,
    reset_dataset: bool,
    frame_skip: int,
) -> None:
    model = ClipModel()
    video_handler = VideoHandler(model)
    milvus_handler = MilvusHandler(collection_name=collection_name, reset_dataset=reset_dataset)
    video_database = VideoDatabase(model, video_handler, milvus_handler, frame_skip=frame_skip)

    # Example usage
    video_database.add_videos_from_folder(video_folder)

    milvus_handler.query(expr="id >= 0")

    video_search = VideoSearch(model, milvus_handler)
    results = video_search.search_by_text(query)
    video_handler.present_query_results(results)
    logging.info("Search results:", results)


if __name__ == "__main__":
    cli()
