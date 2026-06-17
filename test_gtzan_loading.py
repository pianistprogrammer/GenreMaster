import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / 'src'))

try:
    from src.data.gtzan import setup_gtzan
    print("Successfully imported setup_gtzan")
except Exception as e:
    print(f"Error importing setup_gtzan: {e}")
    import traceback
    traceback.print_exc()

def test_gtzan_loading():
    # Use gtzan.yaml config
    config_path = 'configs/gtzan.yaml'
    if not Path(config_path).exists():
        print(f"Config {config_path} not found")
        return

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    print(f"Testing GTZAN loading with:")
    print(f"  audio_dir: {cfg['data']['audio_dir']}")

    try:
        # Check if directory exists
        audio_dir = Path(cfg['data']['audio_dir'])
        if not audio_dir.exists() or not any(audio_dir.glob('**/*.wav')):
             # Try common local path
             potential_paths = [
                 Path("C:/Users/jerem/Documents/Datasets/gtzan/genres_original"),
                 Path("C:/Users/jerem/Documents/Datasets/gtzan"),
             ]
             for p in potential_paths:
                 if p.exists() and any(p.glob('**/*.wav')):
                     audio_dir = p
                     break
             else:
                 print(f"GTZAN audio files not found in search paths")
                 return

        datasets, genre_to_idx = setup_gtzan(
            audio_dir=audio_dir,
            sr=cfg['data'].get('sample_rate', 22050),
            duration=cfg['data'].get('duration', 30.0),
        )
        print(f"Successfully loaded GTZAN datasets!")
        print(f"Train: {len(datasets['train'])}")
        
    except Exception as e:
        print(f"Error loading GTZAN datasets: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gtzan_loading()
