"""Prepare an evaluation-only prompt dataset; official test labels stay separate."""
import argparse
import json
from pathlib import Path
from moe_llm.tokenizer import TextTokenizer, sha256_file


def prepare(source, output, tokenizer, max_seq_len):
 tok=TextTokenizer(tokenizer);root=Path(output)
 if not Path(source).is_file():raise FileNotFoundError(source)
 root.mkdir(parents=True,exist_ok=False);offsets=[];stats={'read':0,'overlong':0,'written':0}
 (root/'train.jsonl').write_bytes(b'');(root/'train.index.json').write_text('[]')
 with Path(source).open() as reader,(root/'val.jsonl').open('xb') as writer:
  for line in reader:
   row=json.loads(line);stats['read']+=1
   messages=row.get('messages') or [{'role':'user','content':row['prompt']}]
   ids=tok.chat(messages,add_generation_prompt=True)[0]
   if len(ids)+1>max_seq_len:stats['overlong']+=1;continue
   record={'messages':messages,'prompt_ids':ids}
   if 'answer' in row:record['answer']=row['answer']
   offsets.append(writer.tell());writer.write((json.dumps(record,ensure_ascii=False)+'\n').encode());stats['written']+=1
 if not offsets:raise ValueError('no evaluation records retained')
 (root/'val.index.json').write_text(json.dumps(offsets))
 m={'format':'moe-lab-posttrain-data-v1','kind':'prompts','evaluation_only':True,'max_seq_len':max_seq_len,'tokenizer_sha256':tok.fingerprint,'sources':{str(source):sha256_file(source)},'stats':stats,'split_records':{'train':0,'val':len(offsets)},'artifacts':{p.name:sha256_file(p) for p in root.iterdir()}}
 (root/'manifest.json').write_text(json.dumps(m,indent=2));print(json.dumps(m,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--tokenizer',required=True);p.add_argument('--max-seq-len',type=int,default=448);a=p.parse_args();prepare(a.input,a.output,a.tokenizer,a.max_seq_len)
