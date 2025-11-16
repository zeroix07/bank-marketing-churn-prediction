import pandas as pd
import numpy as np
import os

# --- Configuration ---
INPUT_FILE = '/Users/pac/Documents/Bank Marketing/dataset/bank/bank-full.csv' 
OUTPUT_DIR = 'new_dataset'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'bank_with_transactions.csv')
DELIMITER = ';'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. Load the original dataset ---
print("Loading original dataset...")
df = pd.read_csv(INPUT_FILE, delimiter=DELIMITER)
# Use the index as the unique Customer ID for linking back to original data
df['Customer_ID'] = df.index  

# --- 2. Initialize list for the new, expanded transactional data ---
synthetic_data = []
end_date = pd.to_datetime('2024-12-31') 

print("Generating synthetic transactional data (Customer IDs will repeat)...")

# --- 3. Loop through clients and create MULTIPLE transactions per customer ---
# The number of transactions is based on the original 'previous' column
for index, row in df.iterrows():
    client_id = row['Customer_ID']
    
    # Use 'previous' as the base for the number of transactions (Frequency). 
    # Ensure at least 2-5 transactions for every client to show repeated history
    base_transactions = row['previous'] if row['previous'] > 0 else np.random.randint(2, 6)
    num_transactions = max(2, int(base_transactions))  # At least 2 transactions per customer
    
    # Use a fraction of 'balance' as the base Transaction Amount (Monetary).
    base_amount = row['balance'] / 10 if row['balance'] > 0 else 100 
    
    for i in range(num_transactions):
        # Generate a unique date for each transaction
        transaction_date = end_date - pd.Timedelta(days=np.random.randint(1, 730))  # Up to 2 years
        
        # Generate a unique amount for each transaction with more variation
        amount = base_amount * np.random.uniform(0.5, 1.5)
        
        # Add a new row for each transaction
        synthetic_data.append({
            'Customer_ID': client_id,
            'Transaction_Date': transaction_date.strftime('%Y-%m-%d'),
            'Transaction_Amount': round(amount, 2),
            'Subscription_Status': row['y'] 
        })
    
    if (index + 1) % 5000 == 0:
        print(f"Processed {index + 1} customers...")

# --- 4. Create the final transactional DataFrame ---
# This DataFrame has REPEATED Customer_IDs showing transaction history
df_transactions = pd.DataFrame(synthetic_data)
df_transactions['Transaction_Date'] = pd.to_datetime(df_transactions['Transaction_Date'])

print(f"\nTotal transactions generated: {len(df_transactions)}")
print(f"Total unique customers: {df_transactions['Customer_ID'].nunique()}")
print(f"Average transactions per customer: {len(df_transactions) / df_transactions['Customer_ID'].nunique():.2f}")

# --- 5. Merge original dataset with transactional data ---
print("\nMerging original dataset with transaction data...")
# This will create repeated rows for each customer based on their transactions
df_merged = df.merge(df_transactions, on='Customer_ID', how='inner')

print(f"\nMerged dataset created:")
print(f"  - Total rows: {len(df_merged)} (with repeated Customer_IDs)")
print(f"  - Unique customers: {df_merged['Customer_ID'].nunique()}")
print(f"  - Total columns: {len(df_merged.columns)}")
print(f"  - New columns: Customer_ID, Transaction_Date, Transaction_Amount, Subscription_Status")

# --- 6. Save the merged dataset ---
try:
    df_merged.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✓ Merged dataset saved to: {OUTPUT_FILE}")
    
    # Show sample of merged data (showing repeated Customer_IDs)
    print("\n" + "="*70)
    print("Sample of MERGED dataset (Customer_IDs repeat for each transaction):")
    print("="*70)
    sample_customer = df_merged['Customer_ID'].iloc[0]
    print(f"\nShowing all transactions for Customer_ID {sample_customer}:")
    sample_cols = ['Customer_ID', 'age', 'job', 'balance', 'Transaction_Date', 'Transaction_Amount', 'y']
    print(df_merged[df_merged['Customer_ID'] == sample_customer][sample_cols].head(10))
    
    print("\n" + "="*70)
    print("First 10 rows of the merged dataset:")
    print("="*70)
    print(df_merged[sample_cols].head(10))
    
except Exception as e:
    print(f"An error occurred while saving the file: {e}")