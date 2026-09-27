import importlib.util
import json
from pathlib import Path
import pytest


def script(name):
    path = Path(__file__).resolve().parents[1] / 'scripts' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sft_import_is_compatible_with_project_chat_format():
    module = script('convert_minimind')
    original = {'conversations': [{'role': 'user', 'content': '你好'},
                {'role': 'assistant', 'content': '<think>private reasoning</think>你好！', 'reasoning_content': 'other reasoning'}]}
    row, reason, stats = module.normalize_sft(original)
    assert reason is None
    assert row['messages'][-1]['content'] == '你好！'
    assert stats['reasoning_fields_omitted'] == stats['thinking_prefixes_removed'] == 1
    from moe_llm.tokenizer import TextTokenizer
    TextTokenizer().chat(row['messages'])
    assert 'private reasoning' in original['conversations'][-1]['content']


@pytest.mark.parametrize('message', [
    {'role': 'tool', 'content': 'result'},
    {'role': 'assistant', 'content': 'call', 'tool_calls': [{'name': 'search'}]},
    {'role': 'assistant', 'content': '<tool_call>{}</tool_call>'},
])
def test_tool_conversation_is_removed_whole(message):
    module = script('convert_minimind')
    row, reason, _ = module.normalize_sft({'conversations': [{'role': 'user', 'content': 'hello'}, message]})
    assert row is None and reason == 'tool_conversation'


def test_import_reports_rejections_and_prevents_overwrite(tmp_path):
    module = script('convert_minimind')
    pretrain, sft = tmp_path / 'pretrain-source.jsonl', tmp_path / 'sft-source.jsonl'
    pretrain.write_text(json.dumps({'text': '原始文本'}) + '\n')
    rows = [{'conversations': [{'role': 'user', 'content': '问'}, {'role': 'assistant', 'content': '答'}]},
            {'conversations': [{'role': 'user', 'content': '问'}]},
            {'conversations': [{'role': 'user', 'content': '问'}, {'role': 'assistant', 'content': '<think>unfinished'}]}]
    sft.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    output = tmp_path / 'normalized'
    report = module.convert(pretrain, sft, output)
    assert report['stats']['sft']['written'] == 1
    assert report['stats']['sft']['skipped_missing_final_answer'] == 1
    assert report['stats']['sft']['skipped_unclosed_thinking'] == 1
    assert module.sha256(output / 'sft.jsonl') == report['outputs']['sft.jsonl']
    with pytest.raises(FileExistsError):
        module.convert(pretrain, sft, output)


def test_download_plan_and_checksum_validation(tmp_path):
    module = script('download_minimind')
    plan = module.plan('mini')
    assert sum(x['bytes'] for x in plan['files'].values()) == 2980244826
    assert len(plan['revision']) == 40
    file = tmp_path / 'data'
    file.write_bytes(b'abc')
    module.verify(file, 3, module.sha256(file))
    with pytest.raises(ValueError):
        module.verify(file, 4, module.sha256(file))


@pytest.mark.parametrize("missing", ["pretrain", "sft"])
def test_missing_input_does_not_create_output(tmp_path, missing):
    module = script('convert_minimind')
    pretrain, sft = tmp_path / 'pretrain.jsonl', tmp_path / 'sft.jsonl'
    pretrain.write_text('{"text":"test"}\n')
    sft.write_text('{}\n')
    (pretrain if missing == "pretrain" else sft).unlink()
    output = tmp_path / 'normalized'
    with pytest.raises(FileNotFoundError, match="Run scripts/download_minimind.py first"):
        module.convert(pretrain, sft, output)
    assert not output.exists()
