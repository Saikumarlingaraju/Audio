"""
Add child speaker data to the dataset and update metadata
Run this if you have child speaker audio files to add
"""
import sys
from pathlib import Path
import pandas as pd
import librosa
from tqdm import tqdm

root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

def scan_for_child_audio(base_dir='data/raw/indian_accents'):
    """
    Scan for child audio files (if they exist in a separate location)
    Update this function to point to your child audio directory
    """
    print(f"\n{'='*80}")
    print("Scanning for Child Speaker Audio")
    print(f"{'='*80}\n")
    
    # Example: If child audio is in 'data/raw/indian_accents_children/'
    child_dir = Path(base_dir + '_children')
    
    if not child_dir.exists():
        print(f"❌ Child audio directory not found: {child_dir}")
        print(f"\nPlease specify where your child audio files are located.")
        print(f"Options:")
        print(f"  1. Create folder: {child_dir}")
        print(f"  2. Or modify this script to point to correct location\n")
        return None
    
    print(f"✓ Found child audio directory: {child_dir}")
    
    # Scan for audio files
    child_files = []
    for state_dir in child_dir.iterdir():
        if state_dir.is_dir():
            state_name = state_dir.name
            for audio_file in state_dir.glob('*.wav'):
                try:
                    # Get audio duration
                    duration = librosa.get_duration(path=str(audio_file))
                    
                    child_files.append({
                        'filepath': str(audio_file.relative_to(child_dir.parent)),
                        'filename': audio_file.name,
                        'speaker_id': f"{state_name}_child_{audio_file.stem}",
                        'native_language': state_name,
                        'age_group': 'child',
                        'utterance_type': 'read',
                        'text': '',
                        'sampling_rate': 16000,
                        'split': 'test',  # Default: use children for testing
                        'duration': duration
                    })
                except Exception as e:
                    print(f"⚠️  Error processing {audio_file.name}: {e}")
    
    return pd.DataFrame(child_files)

def update_metadata_with_children(child_df, output_path='data/raw/indian_accents/metadata_with_children.csv'):
    """
    Merge child data with existing adult metadata
    """
    # Load existing adult data
    adult_df = pd.read_csv('data/raw/indian_accents/metadata_full.csv')
    
    print(f"\n{'='*80}")
    print("Merging Adult and Child Data")
    print(f"{'='*80}\n")
    print(f"Adult samples: {len(adult_df)}")
    print(f"Child samples: {len(child_df)}")
    
    # Combine
    combined_df = pd.concat([adult_df, child_df], ignore_index=True)
    
    print(f"Total samples: {len(combined_df)}")
    print(f"\nAge distribution:")
    print(combined_df['age_group'].value_counts())
    
    # Save
    combined_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved combined metadata: {output_path}")
    
    return combined_df

def main():
    """
    Main function to add child speakers
    """
    print(f"\n{'#'*80}")
    print("# ADD CHILD SPEAKER DATA")
    print(f"{'#'*80}\n")
    
    # Check current data
    current_df = pd.read_csv('data/raw/indian_accents/metadata_full.csv')
    print(f"Current dataset:")
    print(f"  Total samples: {len(current_df)}")
    print(f"  Age groups: {current_df['age_group'].unique()}")
    
    if len(current_df['age_group'].unique()) > 1:
        print(f"\n✓ Dataset already contains multiple age groups!")
        print(current_df['age_group'].value_counts())
        return
    
    print(f"\n⚠️  Dataset currently contains only: {current_df['age_group'].unique()}")
    print(f"\nTo add child speakers:")
    print(f"  1. Place child audio files in: data/raw/indian_accents_children/")
    print(f"  2. Organize by state (same structure as adult data)")
    print(f"  3. Run this script again")
    
    # Try to scan for child audio
    child_df = scan_for_child_audio()
    
    if child_df is not None and len(child_df) > 0:
        print(f"\n✓ Found {len(child_df)} child audio files!")
        
        response = input("\nAdd these to the dataset? (y/n): ").strip().lower()
        if response == 'y':
            combined_df = update_metadata_with_children(child_df)
            
            print(f"\n{'='*80}")
            print("Next Steps:")
            print(f"{'='*80}\n")
            print("1. Update your training scripts to use: metadata_with_children.csv")
            print("2. Run cross-age comparison:")
            print("   python scripts/cross_age_comparison.py")
            print(f"{'='*80}\n")
    else:
        print(f"\n❌ No child audio files found.")
        print(f"\nCurrent metadata shows only adult speakers.")
        print(f"If you have child audio:")
        print(f"  - Add it to: data/raw/indian_accents_children/")
        print(f"  - Or manually update age_group column in metadata")

if __name__ == '__main__':
    main()
