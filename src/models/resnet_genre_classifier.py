"""ResNet-based genre classifier for high-accuracy classification.

Uses pretrained ResNet-18 backbone with SpecAugment for achieving 90%+ accuracy
on the GTZAN dataset.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from torchvision.models import resnet18, ResNet18_Weights


class SpecAugment(nn.Module):
    """
    SpecAugment: A Simple Data Augmentation Method for ASR.

    Applies time and frequency masking to spectrograms.
    Reference: https://arxiv.org/abs/1904.08779
    """

    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
    ):
        super().__init__()
        self.freq_masking = T.FrequencyMasking(freq_mask_param)
        self.time_masking = T.TimeMasking(time_mask_param)
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment to spectrogram [batch, channels, freq, time]."""
        for _ in range(self.n_freq_masks):
            spectrogram = self.freq_masking(spectrogram)
        for _ in range(self.n_time_masks):
            spectrogram = self.time_masking(spectrogram)
        return spectrogram


class ResNetGenreClassifier(nn.Module):
    """
    ResNet-18 based genre classifier with pretrained ImageNet weights.

    Architecture:
    - Log-Mel Spectrogram computation
    - SpecAugment (training only)
    - ResNet-18 backbone (pretrained, modified for 1-channel input)
    - Global Average Pooling
    - Dropout + Classification head

    ~11M parameters with pretrained backbone.
    """

    def __init__(
        self,
        n_genres: int = 10,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        sample_rate: int = 22050,
        pretrained: bool = True,
        dropout: float = 0.5,
        spec_augment: bool = True,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
    ):
        super().__init__()

        self.n_genres = n_genres
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.use_spec_augment = spec_augment

        # Mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )

        # Amplitude to dB
        self.amplitude_to_db = T.AmplitudeToDB()

        # SpecAugment (applied during training only)
        if spec_augment:
            self.spec_augment = SpecAugment(
                freq_mask_param=freq_mask_param,
                time_mask_param=time_mask_param,
                n_freq_masks=n_freq_masks,
                n_time_masks=n_time_masks,
            )
        else:
            self.spec_augment = None

        # ResNet-18 backbone
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
            self.backbone = resnet18(weights=weights)
        else:
            self.backbone = resnet18(weights=None)

        # Modify first conv layer for single-channel spectrogram input
        # Average the pretrained weights across RGB channels
        if pretrained:
            pretrained_conv1 = self.backbone.conv1.weight.data
            new_conv1_weight = pretrained_conv1.mean(dim=1, keepdim=True)
        else:
            new_conv1_weight = None

        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        if pretrained and new_conv1_weight is not None:
            self.backbone.conv1.weight.data = new_conv1_weight

        # Remove original FC layer
        backbone_out_features = self.backbone.fc.in_features  # 512 for ResNet-18
        self.backbone.fc = nn.Identity()

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(backbone_out_features, n_genres),
        )

    def compute_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute log-mel spectrogram from waveform.

        Args:
            waveform: Audio tensor [batch, channels, samples]

        Returns:
            Log-mel spectrogram [batch, 1, n_mels, time]
        """
        # Convert to mono if stereo
        if waveform.dim() == 3 and waveform.shape[1] > 1:
            waveform = torch.mean(waveform, dim=1)
        elif waveform.dim() == 3:
            waveform = waveform.squeeze(1)

        # Ensure mel_transform is on correct device
        if self.mel_transform.mel_scale.fb.device != waveform.device:
            self.mel_transform = self.mel_transform.to(waveform.device)
            self.amplitude_to_db = self.amplitude_to_db.to(waveform.device)

        # Compute mel spectrogram [batch, n_mels, time]
        mel_spec = self.mel_transform(waveform)

        # Convert to dB scale
        log_mel = self.amplitude_to_db(mel_spec)

        # Normalize to roughly [-1, 1] range
        log_mel = (log_mel + 80) / 80

        # Add channel dimension [batch, 1, n_mels, time]
        log_mel = log_mel.unsqueeze(1)

        return log_mel

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: waveform -> genre logits.

        Args:
            waveform: Audio tensor [batch, channels, samples]

        Returns:
            Genre logits [batch, n_genres]
        """
        # Compute spectrogram
        spec = self.compute_spectrogram(waveform)

        # Apply SpecAugment during training
        if self.training and self.spec_augment is not None:
            spec = self.spec_augment(spec)

        # ResNet backbone
        features = self.backbone(spec)

        # Classification
        logits = self.classifier(features)

        return logits

    def predict(self, waveform: torch.Tensor) -> torch.Tensor:
        """Predict genre classes."""
        logits = self(waveform)
        return logits.argmax(dim=1)

    def predict_proba(self, waveform: torch.Tensor) -> torch.Tensor:
        """Predict genre probabilities."""
        logits = self(waveform)
        return F.softmax(logits, dim=1)

    @torch.no_grad()
    def predict_with_segments(
        self,
        waveform: torch.Tensor,
        segment_seconds: float = 3.0,
        hop_seconds: float = 1.5,
        aggregation: str = "soft_vote",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with segment-level voting for improved accuracy.

        Splits audio into overlapping segments, classifies each,
        and aggregates via voting.

        Args:
            waveform: Audio [batch, channels, samples]
            segment_seconds: Duration of each segment
            hop_seconds: Hop between segments (overlap = segment - hop)
            aggregation: 'soft_vote' (average probs) or 'hard_vote' (majority)

        Returns:
            predictions: Genre indices [batch]
            probabilities: Genre probabilities [batch, n_genres]
        """
        self.eval()

        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)

        batch_size, channels, total_samples = waveform.shape
        segment_samples = int(segment_seconds * self.sample_rate)
        hop_samples = int(hop_seconds * self.sample_rate)

        # Extract segments
        segments = []
        for start in range(0, total_samples - segment_samples + 1, hop_samples):
            segment = waveform[:, :, start:start + segment_samples]
            segments.append(segment)

        if not segments:
            segments = [waveform]

        # Process all segments
        all_logits = []
        for segment in segments:
            logits = self(segment)
            all_logits.append(logits)

        # Stack: [n_segments, batch, n_genres]
        all_logits = torch.stack(all_logits, dim=0)

        # Aggregate
        if aggregation == "hard_vote":
            preds = all_logits.argmax(dim=2)  # [n_segments, batch]
            final_pred = torch.mode(preds, dim=0).values
            probs = F.softmax(all_logits.mean(dim=0), dim=1)
        else:  # soft_vote
            probs = F.softmax(all_logits, dim=2).mean(dim=0)
            final_pred = probs.argmax(dim=1)

        return final_pred, probs

    @torch.no_grad()
    def predict_with_tta(
        self,
        waveform: torch.Tensor,
        n_augments: int = 5,
        use_segments: bool = True,
        segment_seconds: float = 3.0,
        hop_seconds: float = 1.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with Test-Time Augmentation (TTA) for maximum accuracy.

        Applies multiple augmentations at inference and averages predictions.

        Args:
            waveform: Audio [batch, channels, samples]
            n_augments: Number of augmented versions to average
            use_segments: Also use segment voting
            segment_seconds: Segment duration if using segments
            hop_seconds: Hop duration if using segments

        Returns:
            predictions: Genre indices [batch]
            probabilities: Genre probabilities [batch, n_genres]
        """
        self.eval()

        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)

        batch_size = waveform.shape[0]
        all_probs = []

        # Original prediction
        if use_segments:
            _, probs = self.predict_with_segments(
                waveform, segment_seconds, hop_seconds
            )
        else:
            probs = self.predict_proba(waveform)
        all_probs.append(probs)

        # Augmented predictions
        for i in range(n_augments - 1):
            aug_waveform = self._apply_tta_augmentation(waveform, i)
            if use_segments:
                _, probs = self.predict_with_segments(
                    aug_waveform, segment_seconds, hop_seconds
                )
            else:
                probs = self.predict_proba(aug_waveform)
            all_probs.append(probs)

        # Average probabilities
        avg_probs = torch.stack(all_probs, dim=0).mean(dim=0)
        predictions = avg_probs.argmax(dim=1)

        return predictions, avg_probs

    def _apply_tta_augmentation(
        self,
        waveform: torch.Tensor,
        aug_idx: int,
    ) -> torch.Tensor:
        """Apply TTA augmentation based on index."""
        # Different augmentations for variety
        if aug_idx == 0:
            # Time shift (roll)
            shift = int(0.1 * waveform.shape[-1])
            return torch.roll(waveform, shifts=shift, dims=-1)
        elif aug_idx == 1:
            # Time shift (other direction)
            shift = int(-0.1 * waveform.shape[-1])
            return torch.roll(waveform, shifts=shift, dims=-1)
        elif aug_idx == 2:
            # Slight gain change
            return waveform * 0.9
        elif aug_idx == 3:
            # Slight gain change (other direction)
            return waveform * 1.1
        else:
            # Small noise
            noise = torch.randn_like(waveform) * 0.005
            return waveform + noise


class MixupAugmentation:
    """
    Mixup: Beyond Empirical Risk Minimization.

    Creates virtual training examples by mixing pairs of samples.
    Reference: https://arxiv.org/abs/1710.09412
    """

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply mixup to batch.

        Args:
            x: Input batch [batch, ...]
            y: Labels [batch] (class indices)

        Returns:
            mixed_x, y_a, y_b, lam
        """
        if self.alpha > 0:
            import numpy as np
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)

        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_resnet_classifier(
    classifier: nn.Module,
    train_loader,
    val_loader,
    num_epochs: int = 100,
    lr: float = 3e-4,
    warmup_epochs: int = 5,
    device: str = "mps",
    save_path: str = "results/genre_classifier_resnet_best.pt",
    patience: int = 20,
    use_mixup: bool = True,
    mixup_alpha: float = 0.4,
    label_smoothing: float = 0.1,
    weight_decay: float = 1e-4,
    gradient_clip: float = 1.0,
    use_trackio: bool = False,
    log_every: int = 10,
    history_path: str = None,
):
    """
    Train ResNet genre classifier with advanced techniques.

    Features:
    - Learning rate warmup
    - Cosine annealing with warm restarts
    - Label smoothing
    - Mixup augmentation
    - Gradient clipping
    """
    import json
    from pathlib import Path
    from tqdm import tqdm

    # Ensure output directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    if history_path is None:
        history_path = Path(save_path).parent / "classifier_resnet_history.json"

    classifier = classifier.to(device)

    # Label smoothing cross-entropy
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # AdamW optimizer
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Learning rate schedulers
    warmup_steps = warmup_epochs * len(train_loader)
    total_steps = num_epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixup augmentation
    mixup = MixupAugmentation(alpha=mixup_alpha) if use_mixup else None

    best_acc = 0.0
    epochs_without_improvement = 0
    global_step = 0

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': [],
        'epoch': [],
        'best_val_acc': 0.0,
        'config': {
            'num_epochs': num_epochs,
            'lr': lr,
            'warmup_epochs': warmup_epochs,
            'patience': patience,
            'use_mixup': use_mixup,
            'mixup_alpha': mixup_alpha,
            'label_smoothing': label_smoothing,
            'weight_decay': weight_decay,
            'device': str(device),
        }
    }

    for epoch in range(num_epochs):
        # Training
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            waveform = batch['waveform'].to(device)
            genre_idx = batch['genre_idx'].to(device)

            # Apply mixup
            if mixup is not None:
                mixed_waveform, y_a, y_b, lam = mixup(waveform, genre_idx)
                logits = classifier(mixed_waveform)
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)

                # For accuracy calculation, use original labels
                pred = logits.argmax(dim=1)
                train_correct += (lam * (pred == y_a).float() +
                                  (1 - lam) * (pred == y_b).float()).sum().item()
            else:
                logits = classifier(waveform)
                loss = criterion(logits, genre_idx)
                pred = logits.argmax(dim=1)
                train_correct += (pred == genre_idx).sum().item()

            train_total += genre_idx.size(0)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), gradient_clip)

            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            global_step += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })

            # Log to Trackio
            if use_trackio and batch_idx % log_every == 0:
                import trackio
                trackio.log({
                    'classifier/train_loss': loss.item(),
                    'classifier/lr': scheduler.get_last_lr()[0],
                    'epoch': epoch,
                }, step=global_step)

        train_acc = 100.0 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        classifier.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                waveform = batch['waveform'].to(device)
                genre_idx = batch['genre_idx'].to(device)

                logits = classifier(waveform)
                loss = criterion(logits, genre_idx)

                val_loss += loss.item()
                pred = logits.argmax(dim=1)
                val_correct += (pred == genre_idx).sum().item()
                val_total += genre_idx.size(0)

        val_acc = 100.0 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%, "
              f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}%")

        # Record history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        # Log epoch metrics to Trackio
        if use_trackio:
            import trackio
            trackio.log({
                'classifier/train_acc': train_acc,
                'classifier/val_acc': val_acc,
                'classifier/train_loss_epoch': avg_train_loss,
                'classifier/val_loss_epoch': avg_val_loss,
                'classifier/best_val_acc': best_acc,
                'epoch': epoch,
            }, step=global_step)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            history['best_val_acc'] = best_acc
            epochs_without_improvement = 0
            torch.save(classifier.state_dict(), save_path)
            print(f"  ✓ New best model! Val Acc={val_acc:.2f}%")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping: no improvement for {patience} epochs")
                break

        # Save history after each epoch
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining complete! Best Val Acc={best_acc:.2f}%")
    return classifier
