import json
import os
from pathlib import Path

def check_wlasl_structure():
    """
    Check WLASL dataset structure and metadata
    """
    print("="*60)
    print("WLASL DATASET STRUCTURE CHECK")
    print("="*60)
    
    # Check videos folder
    videos_path = "data/wlasl_raw/"
    if os.path.exists(videos_path):
        video_files = [f for f in os.listdir(videos_path) 
                      if f.endswith('.mp4')]
        print(f"\n Videos folder: {len(video_files)} MP4 files")
        print(f"   Sample files: {video_files[:5]}")
    else:
        print(f"\n Videos folder not found at: {videos_path}")
        return
    
    # Check metadata JSON
    json_files = {
        'nslt_2000.json': 'data/nslt_2000.json',
        'WLASL_v0.3.json': 'data/WLASL_v0.3.json'
    }
    
    for name, path in json_files.items():
        if os.path.exists(path):
            print(f"\n Found: {name}")
            
            # Load and analyze
            with open(path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                print(f"   Type: List with {len(data)} entries")
                if len(data) > 0:
                    print(f"   First entry keys: {list(data[0].keys())}")
            elif isinstance(data, dict):
                print(f"   Type: Dictionary with keys: {list(data.keys())[:5]}")
        else:
            print(f"\n  Not found: {name}")
    
    # Check class list
    class_list_path = 'data/wlasl_class_list.txt'
    if os.path.exists(class_list_path):
        with open(class_list_path, 'r') as f:
            classes = f.read().strip().split('\n')
        print(f"\n Class list: {len(classes)} words")
        print(f"   Sample words: {classes[:10]}")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("The videos are numbered files (00295.mp4, etc.)")
    print("We'll use the JSON metadata to map videos to words")
    print("during processing.")
    print("="*60)

if __name__ == "__main__":
    check_wlasl_structure()