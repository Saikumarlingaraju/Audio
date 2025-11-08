"""
Create train/dev/test splits with speaker-disjoint and age-based separation.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter


def create_speaker_disjoint_splits(df, train_ratio=0.7, dev_ratio=0.15, 
                                   test_ratio=0.15, random_state=42):
    """
    Create speaker-disjoint train/dev/test splits.
    
    Args:
        df: DataFrame with metadata
        train_ratio: Proportion of speakers for training
        dev_ratio: Proportion of speakers for development
        test_ratio: Proportion of speakers for testing
        random_state: Random seed for reproducibility
    
    Returns:
        train_df, dev_df, test_df
    """
    assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Get unique speakers per language
    speakers_by_lang = df.groupby('native_language')['speaker_id'].unique()
    
    train_speakers = []
    dev_speakers = []
    test_speakers = []
    
    print("\nCreating speaker-disjoint splits per language:")
    print("-" * 80)
    
    for lang, speakers in speakers_by_lang.items():
        speakers = list(speakers)
        n_speakers = len(speakers)
        
        # Calculate split sizes
        n_train = max(1, int(n_speakers * train_ratio))
        n_dev = max(1, int(n_speakers * dev_ratio))
        n_test = n_speakers - n_train - n_dev
        
        # Ensure at least one speaker in test
        if n_test < 1:
            n_test = 1
            n_dev = max(1, n_speakers - n_train - n_test)
        
        # Shuffle and split
        np.random.seed(random_state)
        np.random.shuffle(speakers)
        
        train_speakers.extend(speakers[:n_train])
        dev_speakers.extend(speakers[n_train:n_train+n_dev])
        test_speakers.extend(speakers[n_train+n_dev:])
        
        print(f"{lang:15s}: {n_speakers:3d} speakers -> "
              f"Train: {n_train:3d}, Dev: {n_dev:3d}, Test: {n_test:3d}")
    
    # Create split DataFrames
    train_df = df[df['speaker_id'].isin(train_speakers)].copy()
    dev_df = df[df['speaker_id'].isin(dev_speakers)].copy()
    test_df = df[df['speaker_id'].isin(test_speakers)].copy()
    
    return train_df, dev_df, test_df


def create_age_based_splits(df):
    """
    Create adult-only train/dev sets and children-only test set.
    
    Args:
        df: DataFrame with metadata including 'age_group' column
    
    Returns:
        adult_train_df, adult_dev_df, children_test_df
    """
    # Separate by age group
    adult_df = df[df['age_group'] != 'child'].copy()
    children_df = df[df['age_group'] == 'child'].copy()
    
    print("\nCreating age-based splits:")
    print("-" * 80)
    print(f"Adult samples: {len(adult_df)}")
    print(f"Children samples: {len(children_df)}")
    
    if len(adult_df) == 0:
        print("⚠ Warning: No adult samples found!")
        return None, None, None
    
    if len(children_df) == 0:
        print("⚠ Warning: No children samples found!")
        # Use all adults for train/dev/test
        return create_speaker_disjoint_splits(adult_df)
    
    # Split adults into train/dev (speaker-disjoint)
    adult_speakers = adult_df['speaker_id'].unique()
    train_speakers, dev_speakers = train_test_split(
        adult_speakers, 
        test_size=0.2, 
        random_state=42
    )
    
    adult_train_df = adult_df[adult_df['speaker_id'].isin(train_speakers)].copy()
    adult_dev_df = adult_df[adult_df['speaker_id'].isin(dev_speakers)].copy()
    
    # All children samples for testing
    children_test_df = children_df.copy()
    
    print(f"\nAdult train speakers: {len(train_speakers)}")
    print(f"Adult dev speakers: {len(dev_speakers)}")
    print(f"Children test speakers: {children_df['speaker_id'].nunique()}")
    
    return adult_train_df, adult_dev_df, children_test_df


def create_linguistic_level_splits(df):
    """
    Separate word-level and sentence-level utterances.
    
    Args:
        df: DataFrame with 'utterance_type' column
    
    Returns:
        word_df, sentence_df
    """
    if 'utterance_type' not in df.columns:
        print("⚠ Warning: 'utterance_type' column not found. Skipping linguistic level splits.")
        return None, None
    
    word_df = df[df['utterance_type'] == 'word'].copy()
    sentence_df = df[df['utterance_type'] == 'sentence'].copy()
    
    print("\nLinguistic level distribution:")
    print("-" * 80)
    print(f"Word-level utterances: {len(word_df)}")
    print(f"Sentence-level utterances: {len(sentence_df)}")
    
    return word_df, sentence_df


def analyze_splits(train_df, dev_df, test_df, split_name="Standard"):
    """Print statistics about the splits."""
    print("\n" + "=" * 80)
    print(f"{split_name} Split Statistics")
    print("=" * 80)
    
    for name, split_df in [("Train", train_df), ("Dev", dev_df), ("Test", test_df)]:
        if split_df is None or len(split_df) == 0:
            print(f"\n{name}: No data")
            continue
            
        print(f"\n{name} Set:")
        print(f"  Total samples: {len(split_df)}")
        print(f"  Unique speakers: {split_df['speaker_id'].nunique()}")
        print(f"  Languages: {split_df['native_language'].nunique()}")
        
        # Language distribution
        lang_dist = split_df['native_language'].value_counts()
        print(f"  Language distribution:")
        for lang, count in lang_dist.items():
            pct = 100 * count / len(split_df)
            print(f"    {lang:15s}: {count:5d} samples ({pct:5.1f}%)")
        
        # Duration statistics
        if 'duration' in split_df.columns:
            total_dur = split_df['duration'].sum()
            avg_dur = split_df['duration'].mean()
            print(f"  Total duration: {total_dur:.2f} seconds ({total_dur/3600:.2f} hours)")
            print(f"  Average duration: {avg_dur:.2f} seconds")


def save_splits(output_dir, **splits):
    """
    Save split DataFrames to CSV files.
    
    Args:
        output_dir: Directory to save splits
        **splits: Named DataFrames to save
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("Saving splits...")
    print("=" * 80)
    
    for name, df in splits.items():
        if df is not None and len(df) > 0:
            filepath = output_dir / f"{name}.csv"
            df.to_csv(filepath, index=False)
            print(f"✓ {name}: {len(df)} samples -> {filepath}")
        else:
            print(f"⚠ {name}: Skipped (no data)")


def main():
    parser = argparse.ArgumentParser(
        description="Create train/dev/test splits for NLI dataset"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to metadata CSV file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/splits",
        help="Directory to save split CSV files"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Proportion of speakers for training"
    )
    parser.add_argument(
        "--dev_ratio",
        type=float,
        default=0.15,
        help="Proportion of speakers for development"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Proportion of speakers for testing"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--create_age_splits",
        action="store_true",
        help="Create age-based splits (adults train/dev, children test)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("Dataset Splitting")
    print("=" * 80)
    print(f"Metadata: {args.metadata}")
    print(f"Output directory: {args.output_dir}")
    print(f"Split ratios: {args.train_ratio:.2f} / {args.dev_ratio:.2f} / {args.test_ratio:.2f}")
    print(f"Random seed: {args.random_seed}")
    print("=" * 80)
    
    # Load metadata
    df = pd.read_csv(args.metadata)
    print(f"\n✓ Loaded metadata: {len(df)} samples")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Languages: {df['native_language'].nunique()}")
    print(f"  Speakers: {df['speaker_id'].nunique()}")
    
    # 1. Standard speaker-disjoint splits
    train_df, dev_df, test_df = create_speaker_disjoint_splits(
        df,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_seed
    )
    
    analyze_splits(train_df, dev_df, test_df, "Standard Speaker-Disjoint")
    
    save_splits(
        args.output_dir,
        train=train_df,
        dev=dev_df,
        test=test_df
    )
    
    # 2. Age-based splits (if requested)
    if args.create_age_splits and 'age_group' in df.columns:
        adult_train_df, adult_dev_df, children_test_df = create_age_based_splits(df)
        
        if adult_train_df is not None:
            analyze_splits(adult_train_df, adult_dev_df, children_test_df, 
                          "Age-Based (Adults→Children)")
            
            save_splits(
                args.output_dir,
                adult_train=adult_train_df,
                adult_dev=adult_dev_df,
                children_test=children_test_df
            )
    
    # 3. Linguistic level splits
    if 'utterance_type' in df.columns:
        word_df, sentence_df = create_linguistic_level_splits(df)
        
        if word_df is not None and len(word_df) > 0:
            # Create splits for word-level data
            word_train, word_dev, word_test = create_speaker_disjoint_splits(
                word_df, args.train_ratio, args.dev_ratio, args.test_ratio, args.random_seed
            )
            save_splits(
                args.output_dir,
                word_train=word_train,
                word_dev=word_dev,
                word_test=word_test
            )
        
        if sentence_df is not None and len(sentence_df) > 0:
            # Create splits for sentence-level data
            sent_train, sent_dev, sent_test = create_speaker_disjoint_splits(
                sentence_df, args.train_ratio, args.dev_ratio, args.test_ratio, args.random_seed
            )
            save_splits(
                args.output_dir,
                sentence_train=sent_train,
                sentence_dev=sent_dev,
                sentence_test=sent_test
            )
    
    print("\n" + "=" * 80)
    print("✅ All splits created successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
