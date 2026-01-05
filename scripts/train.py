"""
Main Training Script
Run: python scripts/train.py --config configs/train_config.yaml
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
import argparse
import yaml
from datetime import datetime

from src.data.data_loader import DatasetLoader
from src.data.preprocessor import SimpleTokenizer, RLMDataset
from src.models.rlm import create_rlm, HybridLoss
from src.models.baseline import BaselineComparison, TFIDFBaseline, MeanBaseline
from src.training.trainer import create_trainer, EarlyStopping
from src.visualization.plots import plot_training_curves


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_data(config, data_csv: str = None):
    """Load and prepare datasets"""
    print("\n" + "="*70)
    print("STEP 1: Data Preparation")
    print("="*70)
    
    loader = DatasetLoader(data_dir=config['data']['data_dir'])
    
    # Load dataset: either from provided CSV, data/raw directory, or built-in combined datasets
    if data_csv:
        print(f"Loading dataset from CSV: {data_csv}")
        df = loader.load_from_csv(data_csv)
    else:
        # If there is a data/raw directory with CSVs, prefer that for quick local datasets
        raw_dir = Path(config['data'].get('data_dir', 'data')) / 'raw'
        if raw_dir.exists() and any(raw_dir.glob('*.csv')):
            print(f"Auto-detected raw data in {raw_dir} — loading all CSVs")
            df = loader.load_from_raw_dir(str(raw_dir))
        else:
            # Load combined built-in synthetic datasets
            df = loader.load_combined_dataset()
    
    # Split data
    train_df, val_df, test_df = loader.split_data(
        df,
        test_size=config['data'].get('test_size', 0.2),
        val_size=config['data'].get('val_size', 0.1)
    )
    
    # Print statistics
    stats = loader.get_statistics(df)
    print(f"\nDataset Statistics:")
    print(f"   Total samples: {stats['n_samples']}")
    print(f"   Domains: {stats['n_domains']}")
    print(f"   Target range: [{stats['target_min']:.2f}, {stats['target_max']:.2f}]")
    print(f"   Target mean: {stats['target_mean']:.2f} ± {stats['target_std']:.2f}")
    
    # Save data
    save_dir = Path(config['data']['data_dir']) / 'processed'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(save_dir / 'train.csv', index=False)
    val_df.to_csv(save_dir / 'val.csv', index=False)
    test_df.to_csv(save_dir / 'test.csv', index=False)
    
    print(f"\nSaved processed data to {save_dir}")
    
    return train_df, val_df, test_df


def train_baselines(train_df, test_df, config):
    """Train baseline models for comparison"""
    print("\n" + "="*70)
    print("STEP 2: Training Baseline Models")
    print("="*70)
    
    comparison = BaselineComparison()
    
    texts_train = train_df['text'].tolist()
    targets_train = train_df['target'].values
    texts_test = test_df['text'].tolist()
    targets_test = test_df['target'].values
    
    # Mean baseline
    comparison.add_baseline(
        "Mean Baseline",
        MeanBaseline(),
        texts_train, targets_train,
        texts_test, targets_test
    )
    
    # TF-IDF + Ridge
    comparison.add_baseline(
        "TF-IDF + Ridge",
        TFIDFBaseline(model_type='ridge', max_features=5000),
        texts_train, targets_train,
        texts_test, targets_test
    )
    
    # TF-IDF + Random Forest
    comparison.add_baseline(
        "TF-IDF + Random Forest",
        TFIDFBaseline(model_type='rf', max_features=5000),
        texts_train, targets_train,
        texts_test, targets_test
    )
    
    # Print comparison table
    print(comparison.get_comparison_table())
    
    # Get best baseline
    best_name, best_score = comparison.get_best_model('mae')
    print(f"\nBest Baseline: {best_name} (MAE: {best_score:.4f})")
    
    return comparison, best_score


def train_rlm(train_df, val_df, test_df, config):
    """Train RLM model"""
    print("\n" + "="*70)
    print("STEP 3: Training Regression Language Model")
    print("="*70)
    
    # Build tokenizer
    print("\nBuilding tokenizer...")
    tokenizer = SimpleTokenizer()
    tokenizer.fit(train_df['text'].tolist())
    print(f"   Vocabulary size: {tokenizer.vocab_size}")
    
    # Save tokenizer
    import joblib
    tokenizer_path = Path(config['training']['checkpoint_dir']) / 'tokenizer.pkl'
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(tokenizer, tokenizer_path)
    print(f"   Saved tokenizer to {tokenizer_path}")

    # Save model metadata so demos can reconstruct the architecture reliably
    try:
        import json
        meta = {
            'model_size': config['model'].get('size', config['model'].get('model_size', 'base')),
            'vocab_size': tokenizer.vocab_size
        }
        meta_path = Path(config['training']['checkpoint_dir']) / 'model_meta.json'
        with open(meta_path, 'w') as f:
            json.dump(meta, f)
        print(f"   Saved model metadata to {meta_path}")
    except Exception:
        pass
    
    # Get max_len from config or use default
    max_len = config['data'].get('max_len', 128)
    
    # Create datasets
    train_dataset = RLMDataset(
        list(zip(train_df['text'], train_df['target'])),
        tokenizer,
        max_len=max_len
    )
    
    val_dataset = RLMDataset(
        list(zip(val_df['text'], val_df['target'])),
        tokenizer,
        max_len=max_len
    )
    
    test_dataset = RLMDataset(
        list(zip(test_df['text'], test_df['target'])),
        tokenizer,
        max_len=max_len
    )
    
    # Create data loaders
    num_workers = config['training'].get('num_workers', 0)  # 0 for Windows compatibility
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available() if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available() if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers
    )
    
    # Create model
    print(f"\nBuilding model...")
    model = create_rlm(
        vocab_size=tokenizer.vocab_size,
        model_size=config['model'].get('model_size', 'base')
    )
    
    # Create trainer config
    trainer_config = config['training'].copy()
    trainer_config['n_epochs'] = config['training']['n_epochs']
    
    # Create trainer
    print(f"\nInitializing trainer...")
    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config
    )
    
    # Early stopping
    early_stopping = None
    if config.get('early_stopping', {}).get('patience', 0) > 0:
        early_stopping = EarlyStopping(
            patience=config['early_stopping']['patience'],
            min_delta=config['early_stopping']['min_delta']
        )
    
    # Train
    trainer.train(
        n_epochs=config['training']['n_epochs'],
        early_stopping=early_stopping,
        save_best=True,
        verbose=True
    )
    
    return trainer, model, test_loader, tokenizer


def evaluate_final_model(trainer, test_loader, baseline_score, config):
    """Final evaluation on test set"""
    print("\n" + "="*70)
    print("STEP 4: Final Evaluation")
    print("="*70)
    
    # Load best model
    trainer.load_checkpoint('best_model.pt')
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    trainer.val_loader = test_loader  # Temporarily use test loader
    test_metrics = trainer.validate()
    print(f"\n{'─'*70}")
    print("Final Test Metrics:")
    print(f"{'─'*70}")
    print(f"   MAE:  {test_metrics['mae']:.4f}")
    print(f"   RMSE: {test_metrics['rmse']:.4f}")
    print(f"   R²:   {test_metrics['r2']:.4f}")
    print(f"{'─'*70}")
    
    # Compare with baseline
    if baseline_score < float('inf'):
        improvement = ((baseline_score - test_metrics['mae']) / baseline_score) * 100
        print(f"\nImprovement over best baseline: {improvement:.2f}%")
        
        if improvement > 0:
            print("RLM outperforms baselines!")
        else:
            print("RLM underperforms baselines - consider hyperparameter tuning")
    
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description='Train Regression Language Model')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data-csv', type=str, default=None,
                       help='Path to a CSV file to use as dataset (overrides built-in loaders)')
    parser.add_argument('--skip-baselines', action='store_true',
                       help='Skip baseline training')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    print("\n" + "="*70)
    print("REGRESSION LANGUAGE MODEL - TRAINING PIPELINE")
    print("="*70)
    print(f"Configuration: {args.config}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Set random seeds
    seed = config['training'].get('seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Prepare data (allow overriding dataset with a CSV)
    train_df, val_df, test_df = prepare_data(config, data_csv=args.data_csv)
    
    # Train baselines
    baseline_score = float('inf')
    if not args.skip_baselines:
        comparison, baseline_score = train_baselines(train_df, test_df, config)
    
    # Train RLM
    trainer, model, test_loader, tokenizer = train_rlm(train_df, val_df, test_df, config)
    
    # Final evaluation
    test_metrics = evaluate_final_model(trainer, test_loader, baseline_score, config)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    try:
        plot_training_curves(
            trainer.history,
            save_path=Path(config['training']['log_dir']) / 'training_curves.png'
        )
    except Exception as e:
        print(f"Warning: Could not generate training curves: {e}")
    
    print("\n" + "="*70)
    print("TRAINING PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nResults saved to:")
    print(f"   - Checkpoints: {config['training']['checkpoint_dir']}")
    print(f"   - Logs: {config['training']['log_dir']}")
    print(f"\nTo run the demo:")
    print(f"   streamlit run app_demo/streamlit_app.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

