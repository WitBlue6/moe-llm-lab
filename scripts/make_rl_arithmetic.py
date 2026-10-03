"""Generate clearly marked synthetic arithmetic data for pipeline checks.

This is NOT a general conversation preference dataset or a quality benchmark.
"""
import argparse
import json
from pathlib import Path
import random


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--count',type=int,default=1000)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if not 20 <= args.count <= 20000:
        raise ValueError('count must be between 20 and 20000')
    root=Path(args.output)
    root.mkdir(parents=True,exist_ok=False)
    rng=random.Random(args.seed)
    problems=rng.sample([(a,b) for a in range(200) for b in range(200)],args.count)
    with (root/'preferences.jsonl').open('x') as prefs, (root/'prompts.jsonl').open('x') as prompts:
        for a,b in problems:
            prompt=f'计算 {a}+{b}，只输出一个数字。'
            prefs.write(json.dumps({'prompt':prompt,'chosen':str(a+b),'rejected':str(a+b+1)},ensure_ascii=False)+'\n')
            prompts.write(json.dumps({'prompt':prompt,'answer':str(a+b)},ensure_ascii=False)+'\n')
    (root/'SOURCE.json').write_text(json.dumps({'kind':'synthetic arithmetic','seed':args.seed,'count':args.count,
        'purpose':'pipeline check or narrowly scoped arithmetic experiment; no general-chat claims'},indent=2))


if __name__=='__main__':
    main()
