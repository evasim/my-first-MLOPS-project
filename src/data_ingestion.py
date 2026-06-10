import os
import pandas as pd
from datasets import load_dataset

def download_and_save_ag_news():
    # Create data folder if missing
    os.makedirs("data", exist_ok=True)
    
    print("⏳ Fetching ag_news from Hugging Face...")
    # Load from HF Hub
    dataset = load_dataset("fancyzhx/ag_news")
    
    # Convert training split to a DataFrame
    train_df = pd.DataFrame(dataset["train"])
    
    # Save locally as a raw CSV
    output_path = "data/raw_dataset.csv"
    train_df.to_csv(output_path, index=False)
    print(f"✅ Success! ag_news dataset saved locally to: {output_path}")

if __name__ == "__main__":
    download_and_save_ag_news()
