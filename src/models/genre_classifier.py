"""Simple CNN genre classifier for FMA Small (8 genres).

Lightweight classifier that can be trained on FMA Small's 8 balanced genres
and integrated into GenreMaster for automatic genre detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T


class GenreCNNClassifier(nn.Module):
    """
    Simple CNN classifier for genre classification.

    Architecture:
    - Input: Log-mel spectrogram [batch, 1, time, freq]
    - 4 conv blocks with max pooling
    - Global average pooling
    - Fully connected layers
    - Output: 8 genre logits

    ~500K parameters - lightweight and fast to train
    """

    def __init__(
        self,
        n_genres: int = 8,
        n_mels: int = 128,
        dropout: float = 0.3,
        sample_rate: int = 22050,
    ):
        """
        Initialize genre classifier.

        Args:
            n_genres: Number of genre classes (8 for FMA Small)
            n_mels: Number of mel bands (128 default)
            dropout: Dropout rate for regularization
            sample_rate: Audio sample rate for mel spectrogram
        """
        super().__init__()

        self.n_genres = n_genres
        self.n_mels = n_mels
        self.sample_rate = sample_rate

        # Mel spectrogram transform (registered as buffer, not parameter)
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=n_mels,
        )

        # Convolutional blocks
        # Input: [batch, 1, time, n_mels]
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # /2
            nn.Dropout2d(dropout * 0.5),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # /4
            nn.Dropout2d(dropout * 0.5),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # /8
            nn.Dropout2d(dropout * 0.5),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # /16
            nn.Dropout2d(dropout),
        )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_genres),
        )

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: spectrogram -> genre logits.

        Args:
            spectrogram: Log-mel spectrogram [batch, 1, time, n_mels]
                        or raw audio [batch, channels, samples] (will convert)

        Returns:
            Genre logits [batch, n_genres]
        """
        # If raw audio, convert to spectrogram
        if spectrogram.ndim == 3:
            # Raw audio input - compute log-mel spectrogram
            # Convert to mono
            audio = spectrogram.mean(dim=1)  # [batch, samples]

            # Ensure mel_transform is on the same device
            if self.mel_transform.mel_scale.fb.device != audio.device:
                self.mel_transform = self.mel_transform.to(audio.device)

            # Apply transform
            mel_spec = self.mel_transform(audio)  # [batch, n_mels, time]

            # Convert to log scale
            spectrogram = torch.log(mel_spec + 1e-9)

            # Transpose to [batch, time, n_mels] and add channel dim
            spectrogram = spectrogram.transpose(1, 2).unsqueeze(1)  # [batch, 1, time, n_mels]

        # Convolutional blocks
        x = self.conv1(spectrogram)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Global pooling
        x = self.global_pool(x)
        x = x.flatten(1)

        # Fully connected
        logits = self.fc(x)

        return logits

    def predict(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Predict genre classes.

        Args:
            spectrogram: Input spectrogram or audio

        Returns:
            Predicted genre indices [batch]
        """
        logits = self(spectrogram)
        return logits.argmax(dim=1)

    def predict_proba(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Predict genre probabilities.

        Args:
            spectrogram: Input spectrogram or audio

        Returns:
            Genre probabilities [batch, n_genres]
        """
        logits = self(spectrogram)
        return F.softmax(logits, dim=1)


class GenreMasterWithCNNClassifier(nn.Module):
    """
    GenreMaster with integrated CNN genre classifier.

    Two training modes:
    1. Pre-train classifier separately
    2. Joint training (multi-task learning)
    """

    def __init__(
        self,
        genremaster_model: nn.Module,
        n_genres: int = 8,
        use_pretrained_classifier: bool = False,
        classifier_path: str = None,
    ):
        """
        Initialize GenreMaster with CNN classifier.

        Args:
            genremaster_model: Base GenreMaster model
            n_genres: Number of genres (8 for FMA Small)
            use_pretrained_classifier: If True, load pre-trained classifier
            classifier_path: Path to pre-trained classifier weights
        """
        super().__init__()

        self.genremaster = genremaster_model
        self.classifier = GenreCNNClassifier(n_genres=n_genres)

        # Load pre-trained classifier if provided
        if use_pretrained_classifier and classifier_path:
            self.classifier.load_state_dict(torch.load(classifier_path))
            # Optionally freeze classifier
            # for param in self.classifier.parameters():
            #     param.requires_grad = False

    def forward(
        self,
        waveform: torch.Tensor,
        genre_idx: torch.Tensor = None,
        return_genre_logits: bool = False,
        return_params: bool = False,
    ) -> torch.Tensor | tuple:
        """
        Forward pass with automatic or manual genre.

        Args:
            waveform: Input audio [batch, channels, samples]
            genre_idx: Optional manual genre [batch]
                      If None, predicts from audio
            return_genre_logits: If True, also return genre predictions
            return_params: If True, also return DSP parameters

        Returns:
            Mastered audio [batch, channels, samples]
            (optionally) genre logits [batch, n_genres]
            (optionally) DSP parameters
        """
        # Predict genre if not provided
        if genre_idx is None:
            genre_logits = self.classifier(waveform)
            genre_idx = genre_logits.argmax(dim=1)
        else:
            genre_logits = None

        # Use GenreMaster
        result = self.genremaster(waveform, genre_idx, return_params=return_params)

        # Return based on flags
        if return_params:
            mastered, params = result
            if return_genre_logits:
                return mastered, params, genre_logits
            else:
                return mastered, params
        else:
            mastered = result
            if return_genre_logits:
                return mastered, genre_logits
            else:
                return mastered

    def predict_genre(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Predict genre from audio.

        Args:
            waveform: Audio [batch, channels, samples]

        Returns:
            Genre indices [batch]
        """
        return self.classifier.predict(waveform)


def train_genre_classifier(
    classifier: nn.Module,
    train_loader,
    val_loader,
    num_epochs: int = 20,
    lr: float = 1e-3,
    device: str = "mps",
    save_path: str = "results/genre_classifier_best.pt",
    patience: int = 10,
    use_trackio: bool = False,
    log_every: int = 10,
    history_path: str = None,
):
    """
    Train the genre classifier.

    Args:
        classifier: GenreCNNClassifier model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        num_epochs: Number of epochs
        lr: Learning rate
        device: Device to train on
        save_path: Path to save best model
        patience: Early stopping patience (epochs without improvement)
        use_trackio: Whether to log metrics to Trackio
        log_every: Log metrics every N batches
        history_path: Path to save training history JSON
    """
    import json
    from pathlib import Path
    from tqdm import tqdm

    # Ensure output directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Set default history path if not provided
    if history_path is None:
        history_path = Path(save_path).parent / "classifier_history.json"

    classifier = classifier.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    best_acc = 0.0
    epochs_without_improvement = 0

    # Training history for plotting
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
            'patience': patience,
            'device': str(device),
        }
    }

    for epoch in range(num_epochs):
        # Training
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            waveform = batch['waveform'].to(device)
            genre_idx = batch['genre_idx'].to(device)

            # Forward
            logits = classifier(waveform)
            loss = criterion(logits, genre_idx)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Stats
            train_loss += loss.item()
            pred = logits.argmax(dim=1)
            train_correct += (pred == genre_idx).sum().item()
            train_total += genre_idx.size(0)

            # Log to Trackio
            if use_trackio and batch_idx % log_every == 0:
                import trackio
                step = epoch * len(train_loader) + batch_idx
                trackio.log({
                    'classifier/train_loss': loss.item(),
                    'classifier/lr': optimizer.param_groups[0]['lr'],
                    'epoch': epoch,
                }, step=step)

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
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}%")

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
            }, step=(epoch + 1) * len(train_loader))

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

        scheduler.step()

    print(f"\nTraining complete! Best Val Acc={best_acc:.2f}%")
    return classifier
