import json
from pathlib import Path

from utils.config_handler import agent_conf, chroma_conf, prompts_conf, rag_conf
from utils.path_tool import get_abs_path


def test_project_paths_are_absolute():
    assert Path(get_abs_path("config/chroma.yml")).is_absolute()
    assert Path(get_abs_path("prompts/main_prompt.txt")).is_absolute()


def test_config_keys_exist():
    assert "chat_model_name" in rag_conf
    assert "embedding_model_name" in rag_conf
    assert "collection_name" in chroma_conf
    assert "k" in chroma_conf
    assert "main_prompt_path" in prompts_conf
    assert "external_data_path" in agent_conf


def test_eval_dataset_schema():
    dataset_path = Path(get_abs_path("eval/dataset.jsonl"))
    lines = [x.strip() for x in dataset_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert lines, "eval/dataset.jsonl should not be empty"

    for line in lines:
        item = json.loads(line)
        assert "id" in item
        assert "query" in item
        assert "expected_sources" in item
