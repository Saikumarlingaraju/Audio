"""
Create metadata for your 6 child audio MP3 files
EDIT the state names below to match which state each child is from!
"""
import pandas as pd

# ⚠️ EDIT THIS: Assign the correct state to each child audio file
child_data = [
    {'filename': 'child_aud1.mp3', 'native_language': 'andhra_pradesh'},  # ← Change this!
    {'filename': 'child_aud2.mp3', 'native_language': 'andhra_pradesh'},          # ← Change this!
    {'filename': 'child_aud3.mp3', 'native_language': 'andhra_pradesh'},       # ← Change this!
    {'filename': 'child_aud4.mp3', 'native_language': 'andhra_pradesh'},       # ← Change this!
    {'filename': 'child_aud5.mp3', 'native_language': 'andhra_pradesh'},          # ← Change this!
    {'filename': 'child_aud6.mp3', 'native_language': 'andhra_pradesh'},           # ← Change this!
]

# Valid states (must match your adult training data)
valid_states = ['andhra_pradesh', 'gujrat', 'jharkhand', 'karnataka', 'kerala', 'tamil']

# Create DataFrame
df = pd.DataFrame(child_data)
df['filepath'] = 'child_audios/' + df['filename']
df['age_group'] = 'child'

# Verify states
for state in df['native_language'].unique():
    if state not in valid_states:
        print(f"⚠️  WARNING: '{state}' is not a valid state!")
        print(f"   Valid states: {', '.join(valid_states)}")

# Save
output_file = 'data/raw/child_audios/child_metadata.csv'
df.to_csv(output_file, index=False)

print(f"✓ Created metadata for {len(df)} files")
print(f"  Saved to: {output_file}")
print(f"\nState distribution:")
print(df['native_language'].value_counts())
print(f"\nContents:")
print(df.to_string(index=False))
print("\n" + "="*60)
print("⚠️  IMPORTANT: Edit this script to set correct state for each file!")
print("="*60)
