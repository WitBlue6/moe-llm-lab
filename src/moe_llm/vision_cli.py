"""Optional vision commands, registered without importing optional dependencies."""
import json
from dataclasses import replace
from pathlib import Path


def register(commands):
    fixture = commands.add_parser("vision-fixture", help="create synthetic PNG/chat fixtures for tests only")
    fixture.add_argument("--output", required=True)
    prepare = commands.add_parser("vision-prepare", help="split by image content and prepare visual conversations")
    prepare.add_argument("--input", nargs="+", required=True)
    prepare.add_argument("--image-root", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--tokenizer", default="byte")
    prepare.add_argument("--vision-config", required=True)
    prepare.add_argument("--max-seq-len", type=int, default=512)
    prepare.add_argument("--val-ratio", type=float, default=.05)
    prepare.add_argument("--seed", type=int, default=42)
    train = commands.add_parser("vision-train", help="projector alignment or visual LoRA SFT")
    train.add_argument("--base-checkpoint", required=True)
    train.add_argument("--vision-config", required=True)
    train.add_argument("--train-config", required=True)
    train.add_argument("--data", required=True)
    train.add_argument("--tokenizer", default="byte")
    train.add_argument("--output", required=True)
    train.add_argument("--image-root")
    train.add_argument("--stop-after", type=int)
    train.add_argument("--epochs", type=int, help="positive whole epochs; overrides config epochs and max_steps")
    train.add_argument("--eval-max-batches", type=int,
                       help="limit each validation to this many batches per rank; default: full validation")
    parent = train.add_mutually_exclusive_group()
    parent.add_argument("--init-from")
    parent.add_argument("--resume")
    for name in ("vision-generate", "vision-evaluate"):
        sub = commands.add_parser(name)
        sub.add_argument("--base-checkpoint", required=True)
        sub.add_argument("--checkpoint", required=True, help="visual adapter checkpoint")
        sub.add_argument("--tokenizer", default="byte")
        sub.add_argument("--encoder-path", help="relocate identical frozen encoder files")
        sub.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        if name == "vision-generate":
            prompt = sub.add_mutually_exclusive_group(required=True)
            prompt.add_argument("--prompt")
            prompt.add_argument("--messages-file", help="JSON array of messages ending in user; entire image conversation")
            sub.add_argument("--image", help="omit for the unchanged pure-text path")
            sub.add_argument("--max-new-tokens", type=int, default=64)
            sub.add_argument("--temperature", type=float, default=.8)
            sub.add_argument("--top-k", type=int, default=40)
            sub.add_argument("--seed", type=int, default=42)
        else:
            sub.add_argument("--data", required=True)
            sub.add_argument("--image-root")
            sub.add_argument("--text-data", help="prepared text SFT validation set for exact logit retention check")
            sub.add_argument("--batch-size", type=int, default=1)
            sub.add_argument("--zero-images", action="store_true", help="also evaluate blank-image ablation")
            sub.add_argument("--output", help="optional new JSON results file; never overwrite")


def dispatch(args):
    try:
        import PIL
        import safetensors
    except ImportError as error:
        raise RuntimeError("vision commands require: uv sync --locked --extra vision") from error
    import torch
    from .vision import VisionConfig, ImageProcessor, generate_visual
    from .vision_data import create_visual_fixture, prepare_visual, VisualDataset, visual_chat
    from .vision_training import VisualTrainConfig, train_visual, load_visual_model, evaluate_visual
    from .training import select_device
    if args.command == "vision-fixture":
        print(json.dumps(create_visual_fixture(args.output), indent=2))
    elif args.command == "vision-prepare":
        print(json.dumps(prepare_visual(args.input, args.image_root, args.output, args.tokenizer,
            VisionConfig.load(args.vision_config), args.max_seq_len, args.val_ratio, args.seed), indent=2))
    elif args.command == "vision-train":
        config = VisualTrainConfig.load(args.train_config)
        if args.epochs is not None:
            config = replace(config, epochs=args.epochs)
        train_visual(args.base_checkpoint, VisionConfig.load(args.vision_config),
                     config, args.data, args.tokenizer, args.output,
                     args.init_from, args.resume, args.stop_after, args.image_root, args.eval_max_batches)
    else:
        torch.set_num_threads(1)
        device = select_device(args.device)
        model, tok = load_visual_model(args.base_checkpoint, args.checkpoint, args.tokenizer, device, args.encoder_path)
        if args.command == "vision-generate":
            messages = (json.loads(Path(args.messages_file).read_text()) if args.messages_file else
                        [{"role": "user", "content": args.prompt}])
            pixels = positions = None
            if args.image:
                ids, _, position = visual_chat(tok, messages, model.vision_config.image_tokens, generation=True)
                pixels = ImageProcessor(model.vision_config)(args.image).unsqueeze(0).to(device)
                positions = torch.tensor([position], device=device)
            else:
                ids, _ = tok.chat(messages, add_generation_prompt=True)
            torch.manual_seed(args.seed)
            result = generate_visual(model, torch.tensor([ids], device=device), pixels, positions,
                args.max_new_tokens, args.temperature, args.top_k, tok.eos_id)
            print(tok.decode(result[0, len(ids):].tolist()))
        else:
            data = VisualDataset(args.data, "val", model.vision_config, args.image_root)
            if data.manifest["tokenizer_sha256"] != tok.fingerprint:
                raise ValueError("visual evaluation tokenizer mismatch")
            metrics = evaluate_visual(model, data, args.batch_size, device, "fp32")
            if args.zero_images:
                metrics.update(evaluate_visual(model, data, args.batch_size, device, "fp32", zero_images=True))
            if args.text_data:
                metrics.update(check_text_retention(model, args.base_checkpoint, args.text_data, tok, device, args.batch_size))
            print(json.dumps(metrics, indent=2))
            if args.output:
                with Path(args.output).open("x") as handle:
                    json.dump(metrics, handle, indent=2)


def check_text_retention(model, base_checkpoint, data_path, tokenizer, device, batch_size=1):
    import torch
    from torch.utils.data import DataLoader
    from .model import LanguageModel, ModelConfig, causal_loss_sum
    from .data import TokenDataset, collate
    from .training import load_checkpoint
    data = TokenDataset(data_path, "val")
    if data.manifest["tokenizer_sha256"] != tokenizer.fingerprint:
        raise ValueError("text retention tokenizer mismatch")
    state = load_checkpoint(base_checkpoint)
    base = LanguageModel(ModelConfig(**state["model_config"])).to(device).eval()
    base.load_state_dict(state["model"])
    del state
    model.eval()
    max_error, loss_sum, count_sum = 0., 0., 0
    with torch.inference_mode():
        for batch in DataLoader(data, batch_size=batch_size, collate_fn=collate):
            batch = {k: v.to(device) for k, v in batch.items()}
            original = base(batch["input_ids"], attention_mask=batch["attention_mask"])["logits"]
            retained = model(batch["input_ids"], attention_mask=batch["attention_mask"])["logits"]
            max_error = max(max_error, (original - retained).abs().max().item())
            loss, count = causal_loss_sum(retained, batch["labels"])
            loss_sum += loss.item()
            count_sum += count.item()
    return {"text_max_logit_error": max_error, "text_logits_identical": max_error == 0.,
            "text_val_loss": loss_sum / count_sum, "text_val_tokens": count_sum}
