"""
Clean Checkpoint Utility
- Remove RPN-related parameters from trained checkpoint
- Create inference-only checkpoint for submission
"""
import torch
import argparse
from pathlib import Path


def clean_checkpoint(
    input_path: str,
    output_path: str | None = None,
    keep_config: bool = True,
) -> None:
    """
    Remove RPN-related parameters from checkpoint to create clean inference model.
    
    Args:
        input_path: Path to trained checkpoint (with RPN parameters)
        output_path: Path to save clean checkpoint (default: input_path with '_clean' suffix)
        keep_config: Whether to keep tokenizer_config and model_config in output
    
    Developer log:
    - RPN decoder is only used during training for auxiliary loss
    - Inference model doesn't need RPN parameters
    - This creates a clean checkpoint compatible with baseline Model class
    """
    print(f"📂 Loading checkpoint from: {input_path}")
    checkpoint = torch.load(input_path, map_location="cpu")
    
    # Extract model_state
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model_state = checkpoint["model_state"]
        original_keys = len(model_state)
    elif isinstance(checkpoint, dict):
        model_state = checkpoint
        original_keys = len(model_state)
    else:
        model_state = checkpoint
        original_keys = len(model_state)
    
    print(f"   Original checkpoint keys: {original_keys}")
    
    # Remove RPN-related keys
    clean_state = {
        k: v for k, v in model_state.items()
        if not k.startswith("rpn_")
    }
    
    removed_keys = set(model_state.keys()) - set(clean_state.keys())
    print(f"   Clean checkpoint keys: {len(clean_state)}")
    print(f"   Removed RPN keys: {len(removed_keys)}")
    
    if removed_keys:
        print(f"   Sample removed keys: {list(removed_keys)[:5]}")
    
    # Create clean checkpoint
    clean_checkpoint = {
        "model_state": clean_state,
    }
    
    # Keep config if requested
    if keep_config:
        if "tokenizer_config" in checkpoint:
            clean_checkpoint["tokenizer_config"] = checkpoint["tokenizer_config"]
            print(f"   ✅ Kept tokenizer_config")
        if "model_config" in checkpoint:
            # Remove rpn_vocab from model_config
            model_config = checkpoint["model_config"].copy()
            if "rpn_vocab" in model_config:
                model_config.pop("rpn_vocab")
                print(f"   ✅ Removed rpn_vocab from model_config")
            clean_checkpoint["model_config"] = model_config
            print(f"   ✅ Kept model_config (rpn_vocab removed)")
    
    # Determine output path
    if output_path is None:
        input_path_obj = Path(input_path)
        output_path = str(input_path_obj.parent / f"{input_path_obj.stem}_clean{input_path_obj.suffix}")
    
    # Save clean checkpoint
    print(f"\n💾 Saving clean checkpoint to: {output_path}")
    torch.save(clean_checkpoint, output_path)
    
    # Verify file size
    output_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
    print(f"   ✅ Clean checkpoint saved ({output_size:.2f} MB)")
    print(f"\n🎉 Clean checkpoint created successfully!")
    print(f"   You can now use this checkpoint with Model class (strict=True)")


def main():
    """CLI interface for clean_checkpoint"""
    parser = argparse.ArgumentParser(
        description="Remove RPN parameters from trained checkpoint"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to input checkpoint (with RPN parameters)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to output clean checkpoint (default: input_path with '_clean' suffix)",
    )
    parser.add_argument(
        "--no-config",
        action="store_true",
        help="Don't keep tokenizer_config and model_config in output",
    )
    
    args = parser.parse_args()
    
    clean_checkpoint(
        input_path=args.input,
        output_path=args.output,
        keep_config=not args.no_config,
    )


if __name__ == "__main__":
    main()

