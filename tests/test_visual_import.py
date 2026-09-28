import importlib.util
import io
import json
from pathlib import Path
import pytest


def test_parquet_conversion_preserves_images_and_excludes_text_tools(tmp_path, monkeypatch):
    pa = pytest.importorskip('pyarrow')
    import pyarrow.parquet as pq
    from PIL import Image
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('visual_converter_test', scripts / 'convert_visual_data.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    def blob(size):
        f = io.BytesIO(); Image.new('RGB', size, 'red').save(f, format='PNG'); return f.getvalue()
    image = blob((32, 32))
    question = [{'role': 'user', 'content': '<image> Describe.'}, {'role': 'assistant', 'content': 'A red image.'}]
    tool = [{'role': 'user', 'content': '<image> Describe.'}, {'role': 'assistant', 'content': 'call', 'tool_calls': [{'name': 'x'}]}]
    rows = [{'conversations': json.dumps(question), 'image_bytes': image},
            {'conversations': json.dumps(question), 'image_bytes': image},
            {'conversations': json.dumps(question), 'image_bytes': blob((8, 8))},
            {'conversations': json.dumps(tool), 'image_bytes': image},
            {'conversations': json.dumps([{'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hi'}]), 'image_bytes': blob((8, 8))}]
    source = tmp_path / 'source.parquet'; pq.write_table(pa.Table.from_pylist(rows), source)
    report = module.convert({'align': source, 'sft': source}, tmp_path / 'out')
    assert report['unique_images'] == 1
    assert report['stats']['align']['written'] == 2
    assert report['stats']['align']['contrastive_unique_images'] == 1
    assert report['stats']['sft']['skipped_tool_conversation'] == 1
    out = json.loads((tmp_path / 'out/captions.jsonl').read_text())
    assert (tmp_path / 'out/images' / out['image']).read_bytes() == image
    assert '<image>' not in (tmp_path / 'out/vision-sft.jsonl').read_text()
    with pytest.raises(FileExistsError):
        module.convert({'align': source}, tmp_path / 'out')
