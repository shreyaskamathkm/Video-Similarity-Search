import logging
from pathlib import Path

import click
import yaml
from cloudpathlib import AnyPath

from video_similarity_search.backend.database_handler import MilvusHandler, VideoDatabase
from video_similarity_search.backend.model import model_factory
from video_similarity_search.backend.search import VideoSearch
from video_similarity_search.backend.video_handler import VideoHandler
from video_similarity_search.schema import AppConfig

logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


@cli.command("run_video_similarity")
@click.option(
    "--config-path",
    type=AnyPath,
    help="Path to the config file.",
)
def run_video_similarity(config_path: Path | str) -> None:
    app_config = AppConfig.from_yaml(config_path)

    model = model_factory(
        app_config.model_name, app_config.model_architecture, app_config.model_pretrained
    )
    video_handler = VideoHandler(model)
    milvus_handler = MilvusHandler(
        collection_name=app_config.collection_name,
        reset_dataset=app_config.reset_dataset,
        embedding_size=model.get_embedding_length(),
    )
    video_database = VideoDatabase(
        model, video_handler, milvus_handler, frame_skip=app_config.frame_skip
    )
    video_database.add_videos_from_folder(app_config.video_folder)

    milvus_handler.query(expr="id >= 0")

    video_search = VideoSearch(model, milvus_handler)
    results = video_search.search_by_text(app_config.query)
    video_handler.present_query_results(results)
    logging.info("Search results:", results)


if __name__ == "__main__":
    cli()
