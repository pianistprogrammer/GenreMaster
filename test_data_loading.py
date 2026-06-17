import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / 'src'))

from src.data.fma_small import setup_fma_small

def test_loading():
    config_path = 'configs/fma_classifier.yaml'
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    # Override paths for testing if needed, but I'll update the config file first
    print(f"Testing loading with:")
    print(f"  audio_dir: {cfg['data']['audio_dir']}")
    print(f"  metadata_csv: {cfg['data']['metadata_csv']}")

    try:
        datasets, genre_to_idx = setup_fma_small(
            audio_dir=cfg['data']['audio_dir'],
            metadata_csv=cfg['data']['metadata_csv'],
            sample_rate=cfg['data']['sample_rate'],
            duration=cfg['data']['duration'],
            seed=cfg['experiment']['seed'],
        )
        print(f"Successfully loaded datasets!")
        print(f"Train: {len(datasets['train'])}")
        print(f"Val: {len(datasets['val'])}")
        print(f"Test: {len(datasets['test'])}")
        print(f"Genres: {list(genre_to_idx.keys())}")
        
        # Try to get one item
        item = datasets['train'][0]
        print(f"Waveform shape: {item['waveform'].shape}")
        print(f"Genre index: {item['genre_idx']}")
        
    except Exception as e:
        print(f"Error loading datasets: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_loading()
