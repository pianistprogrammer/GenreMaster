#!/usr/bin/env python3
"""Monitor GenreMaster training progress.

Usage:
    python scripts/monitor_training.py [--pretrain | --main]
"""

import argparse
import time
from pathlib import Path
import json


def monitor_pretrain():
    """Monitor pre-training progress."""
    log_dir = Path("results/logs/pretraining")
    checkpoint_dir = Path("results/pretraining")

    print("=" * 70)
    print("Genre Embedding Pre-training Monitor")
    print("=" * 70)

    # Check for checkpoints
    if checkpoint_dir.exists():
        checkpoints = sorted(checkpoint_dir.glob("*.pt"))
        if checkpoints:
            print(f"\n✓ Checkpoints found: {len(checkpoints)}")
            for ckpt in checkpoints:
                size_mb = ckpt.stat().st_size / (1024 * 1024)
                print(f"  - {ckpt.name} ({size_mb:.1f} MB)")
        else:
            print("\n⏳ No checkpoints yet...")

    # Check log files
    if log_dir.exists():
        logs = sorted(log_dir.glob("*.log"))
        if logs:
            print(f"\n✓ Log files: {len(logs)}")
            latest_log = logs[-1]
            print(f"  Latest: {latest_log.name}")

            # Show last 10 lines
            with open(latest_log) as f:
                lines = f.readlines()
                print("\n  Last 10 lines:")
                for line in lines[-10:]:
                    print(f"    {line.rstrip()}")

    print("\n" + "=" * 70)


def monitor_main():
    """Monitor main training progress."""
    log_dir = Path("results/logs")
    checkpoint_dir = Path("results/checkpoints")
    audio_dir = Path("results/audio_samples")

    print("=" * 70)
    print("GenreMaster Main Training Monitor")
    print("=" * 70)

    # Check for checkpoints
    if checkpoint_dir.exists():
        checkpoints = sorted(checkpoint_dir.glob("*.pt"))
        if checkpoints:
            print(f"\n✓ Checkpoints found: {len(checkpoints)}")
            for ckpt in checkpoints:
                size_mb = ckpt.stat().st_size / (1024 * 1024)
                print(f"  - {ckpt.name} ({size_mb:.1f} MB)")
        else:
            print("\n⏳ No checkpoints yet...")

    # Check audio samples
    if audio_dir.exists():
        samples = sorted(audio_dir.glob("*.wav"))
        if samples:
            print(f"\n✓ Audio samples: {len(samples)}")
            print(f"  Latest: {samples[-1].name}")

    # Check log files
    if log_dir.exists():
        logs = sorted(log_dir.glob("main_training_*.log"))
        if logs:
            print(f"\n✓ Log files: {len(logs)}")
            latest_log = logs[-1]
            print(f"  Latest: {latest_log.name}")

            # Show last 10 lines
            with open(latest_log) as f:
                lines = f.readlines()
                print("\n  Last 10 lines:")
                for line in lines[-10:]:
                    print(f"    {line.rstrip()}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Monitor GenreMaster training")
    parser.add_argument('--pretrain', action='store_true',
                        help='Monitor pre-training')
    parser.add_argument('--main', action='store_true',
                        help='Monitor main training')
    parser.add_argument('--watch', action='store_true',
                        help='Watch mode (refresh every 30s)')
    args = parser.parse_args()

    if not args.pretrain and not args.main:
        print("Specify --pretrain or --main")
        return

    monitor_fn = monitor_pretrain if args.pretrain else monitor_main

    if args.watch:
        print("Watch mode: refreshing every 30 seconds (Ctrl+C to exit)\n")
        try:
            while True:
                monitor_fn()
                time.sleep(30)
                print("\n\n[Refreshing...]\n")
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
    else:
        monitor_fn()


if __name__ == '__main__':
    main()
