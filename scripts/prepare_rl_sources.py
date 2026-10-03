"""Download pinned public RL sources, or convert them to native JSONL schemas."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import urllib.request
from convert_minimind import normalize_sft

REV = '312afb4f76391145c6902f765bb51691c09a12f5'
GSM_REV = '3101c7d5072418e28b9008a6636bde82a006892c'
FILES = {
 'minimind-dpo': {'dpo.jsonl': (f'https://huggingface.co/datasets/jingyaogong/minimind_dataset/resolve/{REV}/dpo.jsonl', 53653322, 'ee934a8a455ccc99d1334d63e1254dd1d64f497fd067cfcbb71e3043f5b46768')},
 'gsm8k': {
  'train.jsonl': (f'https://raw.githubusercontent.com/openai/grade-school-math/{GSM_REV}/grade_school_math/data/train.jsonl',4166206,'17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465'),
  'test.jsonl': (f'https://raw.githubusercontent.com/openai/grade-school-math/{GSM_REV}/grade_school_math/data/test.jsonl',749738,'3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14')}}


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()


def download(dataset, output):
 root=Path(output);root.mkdir(parents=True,exist_ok=True)
 for name,(url,size,digest) in FILES[dataset].items():
  target=root/name
  if not target.exists():
   part=root/(name+'.partial')
   print(f'Downloading {name}: {size/1e6:.1f} MB',flush=True)
   with urllib.request.urlopen(url,timeout=120) as src, part.open('wb') as dest:
    while chunk:=src.read(1024*1024): dest.write(chunk)
   if part.stat().st_size!=size or sha(part)!=digest: raise ValueError(f'checksum mismatch: {part}')
   part.rename(target)
  if target.stat().st_size!=size or sha(target)!=digest: raise ValueError(f'existing file mismatch: {target}')
  print(f'Verified {target}',flush=True)
 info={'dataset':dataset,'revision':REV if dataset=='minimind-dpo' else GSM_REV,'files':FILES[dataset],
       'license_note':'MiniMind is mixed-source (Apache-2.0 / CC-BY-NC-2.0 labels); GSM8K is MIT. Check source cards before redistribution.'}
 record=root/'SOURCES.json'
 if record.exists() and json.loads(record.read_text())!=json.loads(json.dumps(info)): raise ValueError('source manifest mismatch')
 record.write_text(json.dumps(info,indent=2))


def preference(row):
 sides=[]
 for key in ('chosen','rejected'):
  normalized,reason,_=normalize_sft({'messages':row.get(key)})
  if reason:return None,reason
  sides.append(normalized['messages'])
 chosen,rejected=sides
 if chosen[:-1]!=rejected[:-1]:return None,'different_prompt'
 if chosen[-1]['content']==rejected[-1]['content']:return None,'identical_answers'
 return {'messages':chosen[:-1],'chosen':chosen[-1]['content'],'rejected':rejected[-1]['content']},None


def math_prompt(row):
 question=row.get('question');answer=row.get('answer')
 if not isinstance(question,str) or not question.strip() or not isinstance(answer,str):raise ValueError('invalid GSM8K row')
 label=answer.rsplit('####',1)
 if len(label)!=2:raise ValueError('missing final numeric answer')
 number=label[1].strip().replace(',','')
 if not re.fullmatch(r'[+-]?\d+(?:\.\d+)?',number):raise ValueError('unsupported numeric answer')
 return {'prompt':question.strip()+'\nReturn the final numeric answer inside <answer>...</answer>.','answer':number}


def convert(dataset, source, output):
 source=Path(source)
 for name,(_,size,digest) in FILES[dataset].items():
  path=source/name
  if not path.is_file() or path.stat().st_size!=size or sha(path)!=digest:raise ValueError(f'input missing or checksum mismatch: {path}')
 root=Path(output);root.mkdir(parents=True,exist_ok=False);counts=Counter()
 if dataset=='minimind-dpo':
  with (source/'dpo.jsonl').open() as reader, (root/'preferences.jsonl').open('x') as prefs, (root/'prompts.jsonl').open('x') as prompts:
   seen=set()
   for line in reader:
    counts['read']+=1;pair,reason=preference(json.loads(line))
    if reason:counts['skipped_'+reason]+=1;continue
    prefs.write(json.dumps(pair,ensure_ascii=False)+'\n');counts['preferences']+=1
    key=json.dumps(pair['messages'],sort_keys=True,ensure_ascii=False)
    if key not in seen:
     seen.add(key);prompts.write(json.dumps({'messages':pair['messages']},ensure_ascii=False)+'\n');counts['prompts']+=1
 else:
  for split in ('train','test'):
   with (source/f'{split}.jsonl').open() as reader,(root/f'{split}-prompts.jsonl').open('x') as writer:
    for line in reader:
     writer.write(json.dumps(math_prompt(json.loads(line)),ensure_ascii=False)+'\n');counts[split]+=1
 report={'dataset':dataset,'counts':dict(counts),'sources':{name:sha(source/name) for name in FILES[dataset]},'outputs':{p.name:sha(p) for p in root.iterdir()},'converter_sha256':sha(__file__),
 'policy':'DPO: require shared prompt, exclude tools, remove leading closed think blocks, omit separate reasoning; no fabricated preferences. GSM8K: question only in prompt, final numeric label separate; official test is never merged into train.'}
 (root/'import-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['download','convert']);p.add_argument('--dataset',choices=FILES,required=True);p.add_argument('--source');p.add_argument('--output',required=True)
 a=p.parse_args()
 if a.action=='download':download(a.dataset,a.output)
 else:
  if not a.source:p.error('convert requires --source')
  convert(a.dataset,a.source,a.output)
if __name__=='__main__':main()
