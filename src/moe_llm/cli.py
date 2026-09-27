import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from .data import prepare, TokenDataset
from .model import ModelConfig, LanguageModel, generate
from .tokenizer import TextTokenizer, train_bpe
from .training import TrainConfig, train, load_checkpoint, select_device, evaluate


def main(argv=None):
    parser = argparse.ArgumentParser(description="Readable MoE pretraining and SFT lab")
    commands = parser.add_subparsers(dest="command", required=True)
    estimate = commands.add_parser("inspect", help="validate config and calculate parameters without allocating weights")
    estimate.add_argument("--model-config", required=True)
    tokenizer = commands.add_parser("tokenizer", help="train byte-level BPE on training-only JSONL")
    tokenizer.add_argument("--input", nargs="+", required=True)
    tokenizer.add_argument("--output", required=True)
    tokenizer.add_argument("--vocab-size", type=int, default=16384)
    tokenizer.add_argument("--val-ratio", type=float, default=0.05)
    tokenizer.add_argument("--seed", type=int, default=42)
    prepare_parser = commands.add_parser("prepare", help="deduplicate, split documents and pre-tokenize")
    prepare_parser.add_argument("--input", nargs="+", required=True)
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.add_argument("--tokenizer", default="byte")
    prepare_parser.add_argument("--stage", choices=["pretrain", "sft"], required=True)
    prepare_parser.add_argument("--max-seq-len", type=int, default=512)
    prepare_parser.add_argument("--val-ratio", type=float, default=0.05)
    prepare_parser.add_argument("--seed", type=int, default=42)
    training = commands.add_parser("train", help="pretrain or SFT; also launched through torchrun")
    training.add_argument("--model-config", required=True)
    training.add_argument("--train-config", required=True)
    training.add_argument("--data", required=True)
    training.add_argument("--tokenizer", default="byte")
    training.add_argument("--output", required=True)
    parent = training.add_mutually_exclusive_group()
    parent.add_argument("--init-from")
    parent.add_argument("--resume")
    training.add_argument("--stop-after", type=int, help="stop early without changing the LR scheduling horizon")
    training.add_argument("--eval-max-batches", type=int, help="limit each validation to this many batches per rank; default: full validation")
    for command in ("generate", "evaluate"):
        sub = commands.add_parser(command)
        sub.add_argument("--checkpoint", required=True)
        sub.add_argument("--tokenizer", default="byte")
        sub.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
        if command == "generate":
            sub.add_argument("--prompt", required=True)
            sub.add_argument("--chat", action="store_true", help="format a user turn and assistant prefix for SFT")
            sub.add_argument("--max-new-tokens", type=int, default=64)
            sub.add_argument("--temperature", type=float, default=0.8)
            sub.add_argument("--top-k", type=int, default=40)
            sub.add_argument("--seed", type=int, default=42)
        else:
            sub.add_argument("--data", required=True)
            sub.add_argument("--batch-size", type=int, default=1)
    from .vision_cli import register
    register(commands)
    doctor = commands.add_parser("doctor", help="read-only Python/CUDA/GPU diagnostics")
    doctor.add_argument("--output", help="save diagnostics to a new JSON file")
    configure = commands.add_parser("configure", help="create a model config with the actual tokenizer vocabulary")
    configure.add_argument("--model-config", required=True)
    configure.add_argument("--tokenizer", required=True)
    configure.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command.startswith("vision-"):
        from .vision_cli import dispatch
        return dispatch(args)
    if args.command in ("doctor", "configure"):
        from .diagnostics import doctor, configure
        if args.command == "doctor":
            doctor(args.output)
            return None
        return configure(args.model_config, args.tokenizer, args.output)
    if args.command == "inspect":
        config = ModelConfig.load(args.model_config)
        print(json.dumps({"config": asdict(config), "parameters": config.parameter_counts()}, indent=2))
    elif args.command == "tokenizer":
        print(json.dumps(train_bpe(args.input, args.output, args.vocab_size, args.val_ratio, args.seed), indent=2))
    elif args.command == "prepare":
        print(json.dumps(prepare(args.input, args.output, args.tokenizer, args.stage,
                                 args.max_seq_len, args.val_ratio, args.seed), indent=2))
    elif args.command == "train":
        train(ModelConfig.load(args.model_config), TrainConfig.load(args.train_config),
              args.data, args.tokenizer, args.output, args.init_from, args.resume, args.stop_after, args.eval_max_batches)
    else:
        torch.set_num_threads(1)
        tok = TextTokenizer(args.tokenizer)
        state = load_checkpoint(args.checkpoint)
        if tok.fingerprint != state["provenance"]["tokenizer_sha256"]:
            raise ValueError("checkpoint tokenizer fingerprint mismatch")
        device = select_device(args.device)
        config = ModelConfig(**state["model_config"])
        model = LanguageModel(config).to(device)
        model.load_state_dict(state["model"])
        model.eval()
        if args.command == "generate":
            torch.manual_seed(args.seed)
            ids = (tok.chat([{"role": "user", "content": args.prompt}], True)[0] if args.chat
                   else [tok.bos_id, *tok.encode(args.prompt)])
            if len(ids) >= config.max_seq_len:
                raise ValueError("prompt leaves no room for generation")
            result = generate(model, torch.tensor([ids], device=device), args.max_new_tokens,
                              args.temperature, args.top_k, tok.eos_id)
            print(tok.decode(result[0, len(ids):].tolist()))
        else:
            dataset = TokenDataset(args.data, "val")
            if dataset.manifest["tokenizer_sha256"] != tok.fingerprint:
                raise ValueError("evaluation tokenizer mismatch")
            print(json.dumps(evaluate(model, dataset, args.batch_size, device, "fp32"), indent=2))


if __name__ == "__main__":
    main()
