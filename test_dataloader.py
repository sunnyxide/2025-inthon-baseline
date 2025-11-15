"""Test script for enhanced dataloader"""
from dataloader import ArithmeticDataset, get_dataloader
from collections import Counter

def test_dataset():
    """Test dataset generation and distribution"""
    print("=" * 60)
    print("Testing Enhanced ArithmeticDataset")
    print("=" * 60)
    
    # Create small test dataset
    dataset = ArithmeticDataset(
        num_samples=1000,
        phase=4,  # 4-5 digits
        seed=42,
        mode="train",
        enable_augmentation=True,
    )
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Phase: {dataset.phase}")
    
    # Sample some data
    categories = []
    digit_lengths = []
    output_6digit_count = 0
    samples = []
    
    for i in range(min(100, len(dataset))):
        sample = dataset[i]
        categories.append(sample["meta"]["category"])
        digit_lengths.append(sample["meta"]["digit_len"])
        if sample["meta"]["output_6digit"]:
            output_6digit_count += 1
        samples.append(sample)
    
    # Print statistics
    print("\n" + "-" * 60)
    print("Category Distribution (first 100 samples):")
    print("-" * 60)
    cat_counts = Counter(categories)
    for cat, count in cat_counts.most_common():
        print(f"  {cat:25s}: {count:3d} ({count/len(categories)*100:.1f}%)")
    
    print("\n" + "-" * 60)
    print("Digit Length Distribution:")
    print("-" * 60)
    digit_counts = Counter(digit_lengths)
    for digit_len, count in sorted(digit_counts.items()):
        print(f"  {digit_len} digits: {count:3d} ({count/len(digit_lengths)*100:.1f}%)")
    
    print("\n" + "-" * 60)
    print("Output 6+ Digit Ratio:")
    print("-" * 60)
    print(f"  {output_6digit_count}/{len(samples)} ({output_6digit_count/len(samples)*100:.1f}%)")
    
    # Print sample examples
    print("\n" + "-" * 60)
    print("Sample Examples:")
    print("-" * 60)
    for i, sample in enumerate(samples[:10]):
        print(f"  [{i+1}] {sample['input_text']:20s} → {sample['target_text']:10s} "
              f"[{sample['meta']['category']:20s}, {sample['meta']['digit_len']}dig, "
              f"6+dig: {sample['meta']['output_6digit']}]")
    
    # Test DataLoader
    print("\n" + "=" * 60)
    print("Testing DataLoader with validation")
    print("=" * 60)
    
    dataloader = get_dataloader(
        dataset,
        batch_size=32,
        num_workers=0,
        mode="train",
    )
    
    # Get one batch
    batch = next(iter(dataloader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Batch size: {len(batch['input_text'])}")
    print(f"\nFirst 5 samples in batch:")
    for i in range(min(5, len(batch['input_text']))):
        print(f"  [{i+1}] {batch['input_text'][i]:20s} → {batch['target_text'][i]}")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_dataset()

