import importlib.util
import json
from pathlib import Path
import sys
import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
from prepare_rl_sources import preference, math_prompt
from prepare_rl_test import prepare
from moe_llm.posttrain_data import PosttrainDataset


def test_pair_shared_context_and_clean_answer():
 row={'chosen':[{'role':'user','content':'Q'},{'role':'assistant','content':'<think>private</think> good'}], 'rejected':[{'role':'user','content':'Q'},{'role':'assistant','content':'bad'}]}
 pair,reason=preference(row)
 assert reason is None and pair['chosen']=='good'
 assert pair['messages']==[{'role':'user','content':'Q'}]
 row['rejected'][0]['content']='different'
 assert preference(row)[1]=='different_prompt'
 row['rejected'][0]={'role':'user','content':'Q','tool_calls':[{}]}
 assert preference(row)[1]=='tool_conversation'


def test_numeric_label_not_visible():
 row=math_prompt({'question':'How many apples?', 'answer':'secret reasoning\n#### 1,234'})
 assert row['answer']=='1234'
 assert 'secret' not in row['prompt'] and '1234' not in row['prompt']
 with pytest.raises(ValueError):math_prompt({'question':'Q','answer':'no final answer'})


def test_official_test_evaluation_only(tmp_path):
 source=tmp_path/'raw.jsonl';source.write_text(json.dumps({'prompt':'Q','answer':'42'})+'\n')
 out=tmp_path/'prepared';prepare(source,out,'byte',448)
 assert len(PosttrainDataset(out,'train'))==0
 dataset=PosttrainDataset(out,'val');assert len(dataset)==1 and dataset[0]['answer']=='42'
 assert dataset[0]['messages']==[{'role':'user','content':'Q'}]
 with pytest.raises(FileExistsError):prepare(source,out,'byte',448)
