"""
Data Loader for Multiple Real-World Datasets
Supports: Yelp Reviews, Financial News, Medical Notes, Real Estate
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import requests
import zipfile
import io
from sklearn.model_selection import train_test_split


class DatasetLoader:
    """Unified interface for loading various regression datasets"""
    
    def __init__(self, data_dir: str = "data/"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_yelp_reviews(self, n_samples: int = 10000) -> pd.DataFrame:
        """
        Load Yelp reviews with star ratings (1-5)
        Target: Predict star rating from review text
        """
        print("📥 Loading Yelp Reviews dataset...")
        
        # Using Yelp Academic Dataset or similar
        # For demo, we'll create realistic synthetic data
        # In production, replace with actual Yelp API or dataset
        
        templates = [
            ("Amazing food and service! Highly recommend.", 5.0),
            ("Pretty good experience overall.", 4.0),
            ("It was okay, nothing special.", 3.0),
            ("Disappointed with the quality.", 2.0),
            ("Terrible experience, won't return.", 1.0),
            ("Absolutely loved it! Best {item} ever!", 5.0),
            ("Great {item}, will definitely come back.", 4.5),
            ("Good {item} but a bit pricey.", 3.5),
            ("The {item} was mediocre at best.", 2.5),
            ("Worst {item} I've ever had.", 1.5),
        ]
        
        items = ["pizza", "burger", "sushi", "pasta", "steak", 
                "salad", "dessert", "coffee", "breakfast", "dinner"]
        
        data = []
        np.random.seed(42)
        
        for _ in range(n_samples):
            template, base_rating = templates[np.random.randint(len(templates))]
            if "{item}" in template:
                text = template.format(item=np.random.choice(items))
            else:
                text = template
            
            # Add noise to ratings
            rating = base_rating + np.random.normal(0, 0.2)
            rating = np.clip(rating, 1.0, 5.0)
            
            data.append({
                'text': text,
                'target': rating,
                'domain': 'restaurant_reviews'
            })
        
        df = pd.DataFrame(data)
        print(f"✅ Loaded {len(df)} Yelp reviews")
        return df
    
    def load_financial_sentiment(self, n_samples: int = 5000) -> pd.DataFrame:
        """
        Load financial news with stock price change
        Target: Predict % price change from news headline
        """
        print("📥 Loading Financial Sentiment dataset...")
        
        positive_templates = [
            ("{company} reports record quarterly earnings", lambda: np.random.uniform(2, 8)),
            ("{company} announces major partnership", lambda: np.random.uniform(1.5, 5)),
            ("{company} beats analyst expectations", lambda: np.random.uniform(2, 6)),
            ("Strong growth forecast for {company}", lambda: np.random.uniform(1, 4)),
            ("{company} launches innovative product", lambda: np.random.uniform(1.5, 5.5)),
        ]
        
        negative_templates = [
            ("{company} faces regulatory scrutiny", lambda: np.random.uniform(-6, -2)),
            ("{company} misses earnings targets", lambda: np.random.uniform(-5, -1.5)),
            ("CEO of {company} steps down unexpectedly", lambda: np.random.uniform(-4, -1)),
            ("{company} recalls products", lambda: np.random.uniform(-3.5, -1)),
            ("Lawsuit filed against {company}", lambda: np.random.uniform(-3, -0.5)),
        ]
        
        neutral_templates = [
            ("{company} maintains steady performance", lambda: np.random.uniform(-0.5, 0.5)),
            ("{company} announces routine updates", lambda: np.random.uniform(-0.3, 0.3)),
        ]
        
        companies = ["Apple", "Microsoft", "Tesla", "Amazon", "Google", 
                    "Meta", "Netflix", "NVIDIA", "Intel", "AMD"]
        
        all_templates = positive_templates + negative_templates + neutral_templates
        
        data = []
        np.random.seed(42)
        
        for _ in range(n_samples):
            template, price_fn = all_templates[np.random.randint(len(all_templates))]
            company = np.random.choice(companies)
            text = template.format(company=company)
            price_change = price_fn()
            
            data.append({
                'text': text,
                'target': price_change,
                'domain': 'financial_news'
            })
        
        df = pd.DataFrame(data)
        print(f"✅ Loaded {len(df)} financial news items")
        return df
    
    def load_real_estate(self, n_samples: int = 3000) -> pd.DataFrame:
        """
        Load real estate descriptions with prices
        Target: Predict price from property description
        """
        print("📥 Loading Real Estate dataset...")
        
        templates = [
            ("Spacious {beds}BR {baths}BA {property_type} in {location}", 
             lambda b, ba: 200000 + b * 50000 + ba * 30000 + np.random.normal(0, 20000)),
            ("Cozy {beds}BR {property_type} with {feature}", 
             lambda b, ba: 180000 + b * 45000 + np.random.normal(0, 15000)),
            ("Luxurious {beds}BR {baths}BA {property_type} featuring {feature}", 
             lambda b, ba: 400000 + b * 80000 + ba * 40000 + np.random.normal(0, 30000)),
            ("Modern {beds}BR {property_type} near {location}", 
             lambda b, ba: 250000 + b * 55000 + np.random.normal(0, 18000)),
        ]
        
        property_types = ["apartment", "house", "condo", "townhouse"]
        locations = ["downtown", "suburbs", "waterfront", "city center"]
        features = ["hardwood floors", "granite countertops", "city views", 
                   "updated kitchen", "large backyard", "pool"]
        
        data = []
        np.random.seed(42)
        
        for _ in range(n_samples):
            template, price_fn = templates[np.random.randint(len(templates))]
            beds = np.random.randint(1, 6)
            baths = np.random.randint(1, 4)
            
            text = template.format(
                beds=beds,
                baths=baths,
                property_type=np.random.choice(property_types),
                location=np.random.choice(locations),
                feature=np.random.choice(features)
            )
            
            price = price_fn(beds, baths)
            price = max(100000, price)  # Floor price
            
            data.append({
                'text': text,
                'target': price,
                'domain': 'real_estate'
            })
        
        df = pd.DataFrame(data)
        print(f"✅ Loaded {len(df)} real estate listings")
        return df
    
    def load_combined_dataset(self) -> pd.DataFrame:
        """Load and combine all datasets"""
        print("🔄 Loading combined multi-domain dataset...\n")
        
        yelp_df = self.load_yelp_reviews(n_samples=8000)
        finance_df = self.load_financial_sentiment(n_samples=4000)
        estate_df = self.load_real_estate(n_samples=3000)
        
        combined = pd.concat([yelp_df, finance_df, estate_df], ignore_index=True)
        combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\n✅ Combined dataset: {len(combined)} samples across {combined['domain'].nunique()} domains")
        print(f"   Domains: {combined['domain'].unique().tolist()}")
        
        return combined

    def load_from_csv(
        self,
        csv_path: str,
        text_col: str = 'text',
        target_col: str = 'target',
        domain_col: Optional[str] = None,
        n_samples: Optional[int] = None,
        sample_frac: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Load a dataset from a local CSV file. The CSV must contain at least
        a text column and a target column. Optionally a domain column can be
        provided to preserve domain labels.

        Args:
            csv_path: Path to CSV file
            text_col: Column name for text
            target_col: Column name for target value
            domain_col: Column name for domain labels (optional)
            n_samples: If set, randomly sample this many rows
            sample_frac: If set, randomly sample this fraction of rows
        Returns:
            pd.DataFrame with `text`, `target`, `domain` columns
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(path)

        if text_col not in df.columns or target_col not in df.columns:
            raise ValueError(f"CSV must contain columns '{text_col}' and '{target_col}'")

        df = df.rename(columns={text_col: 'text', target_col: 'target'})

        if domain_col and domain_col in df.columns:
            df = df.rename(columns={domain_col: 'domain'})
        else:
            df['domain'] = 'custom'

        # Optional sampling
        if n_samples is not None:
            df = df.sample(n=min(n_samples, len(df)), random_state=42).reset_index(drop=True)
        elif sample_frac is not None:
            df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)

        print(f"Loaded {len(df)} samples from {csv_path}")
        return df

    def load_from_raw_dir(self, raw_dir: str = None, pattern: str = "*.csv") -> pd.DataFrame:
        """
        Load and concatenate all CSV files from a raw data directory. Useful when
        you drop multiple source CSVs under `data/raw/`.
        """
        raw_dir = raw_dir or str(self.data_dir / 'raw')
        p = Path(raw_dir)
        if not p.exists():
            raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

        files = list(p.glob(pattern))
        if len(files) == 0:
            raise FileNotFoundError(f"No files matching {pattern} in {raw_dir}")

        dfs = []
        for f in files:
            try:
                df = pd.read_csv(f)
                # Attempt to standardize common column names
                if 'text' not in df.columns and 'content' in df.columns:
                    df = df.rename(columns={'content': 'text'})
                if 'target' not in df.columns and 'label' in df.columns:
                    df = df.rename(columns={'label': 'target'})
                if 'text' in df.columns and 'target' in df.columns:
                    if 'domain' not in df.columns:
                        df['domain'] = f.stem
                    dfs.append(df[['text', 'target', 'domain']])
                else:
                    print(f"Skipping {f} - missing required columns")
            except Exception as e:
                print(f"Failed to read {f}: {e}")

        if len(dfs) == 0:
            raise RuntimeError("No valid CSVs found in raw directory")

        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Loaded combined raw dataset: {len(combined)} samples from {len(dfs)} files")
        return combined
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, 
                   val_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/val/test sets"""
        
        # First split: train+val vs test
        train_val, test = train_test_split(
            df, test_size=test_size, random_state=42, stratify=df['domain']
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val, test_size=val_ratio, random_state=42, stratify=train_val['domain']
        )
        
        print(f"\nData Split:")
        print(f"   Train: {len(train)} samples ({len(train)/len(df)*100:.1f}% )")
        print(f"   Val:   {len(val)} samples ({len(val)/len(df)*100:.1f}% )")
        print(f"   Test:  {len(test)} samples ({len(test)/len(df)*100:.1f}% )")
        
        return train, val, test
    
    def get_statistics(self, df: pd.DataFrame) -> Dict:
        """Get dataset statistics"""
        stats = {
            'n_samples': len(df),
            'n_domains': df['domain'].nunique(),
            'target_mean': df['target'].mean(),
            'target_std': df['target'].std(),
            'target_min': df['target'].min(),
            'target_max': df['target'].max(),
            'avg_text_length': df['text'].str.len().mean(),
            'domain_distribution': df['domain'].value_counts().to_dict()
        }
        return stats


# Example usage
if __name__ == "__main__":
    loader = DatasetLoader()
    
    # Load individual datasets
    # yelp_df = loader.load_yelp_reviews()
    # finance_df = loader.load_financial_sentiment()
    # estate_df = loader.load_real_estate()
    
    # Load combined dataset
    df = loader.load_combined_dataset()
    
    # Split data
    train_df, val_df, test_df = loader.split_data(df)
    
    # Get statistics
    stats = loader.get_statistics(df)
    print(f"\nDataset Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Save processed data
    (Path('data/processed')).mkdir(parents=True, exist_ok=True)
    train_df.to_csv('data/processed/train.csv', index=False)
    val_df.to_csv('data/processed/val.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)
    print(f"\n💾 Saved processed data to data/processed/")
