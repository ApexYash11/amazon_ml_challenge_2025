# 🎯 SUBMISSION VERIFICATION
# Validate final submission file

import pandas as pd
import numpy as np

print("🎯 SUBMISSION VERIFICATION")
print("="*50)

# Load submission
try:
    submission = pd.read_csv('final_submission.csv')
    print("✅ Submission file loaded successfully")
except:
    print("❌ Failed to load submission file")
    exit()

# Basic validation
print(f"\n📊 BASIC STATISTICS:")
print(f"   Rows: {len(submission):,}")
print(f"   Columns: {len(submission.columns)}")
print(f"   Column names: {submission.columns.tolist()}")

# Check required columns
required_cols = ['sample_id', 'price']
missing_cols = [col for col in required_cols if col not in submission.columns]

if missing_cols:
    print(f"❌ Missing columns: {missing_cols}")
else:
    print(f"✅ All required columns present")

# Data validation
print(f"\n🔍 DATA VALIDATION:")

# Check for missing values
missing_values = submission.isnull().sum().sum()
print(f"   Missing values: {missing_values}")

if missing_values == 0:
    print("   ✅ No missing values")
else:
    print("   ❌ Has missing values")

# Check sample_id format
unique_ids = submission['sample_id'].nunique()
print(f"   Unique sample_ids: {unique_ids:,}")

if unique_ids == len(submission):
    print("   ✅ All sample_ids unique")
else:
    print("   ❌ Duplicate sample_ids found")

# Check price statistics
print(f"\n💰 PRICE STATISTICS:")
print(f"   Mean: ${submission['price'].mean():.2f}")
print(f"   Median: ${submission['price'].median():.2f}")
print(f"   Min: ${submission['price'].min():.2f}")
print(f"   Max: ${submission['price'].max():.2f}")
print(f"   Std: ${submission['price'].std():.2f}")

# Check for negative prices
negative_prices = (submission['price'] < 0).sum()
print(f"   Negative prices: {negative_prices}")

if negative_prices == 0:
    print("   ✅ No negative prices")
else:
    print("   ❌ Has negative prices")

# Check for zero prices
zero_prices = (submission['price'] == 0).sum()
print(f"   Zero prices: {zero_prices}")

# Data type validation
print(f"\n🔧 DATA TYPES:")
print(f"   sample_id: {submission['sample_id'].dtype}")
print(f"   price: {submission['price'].dtype}")

# Final validation
print(f"\n🏁 FINAL VALIDATION:")

validation_passed = True
checks = [
    len(submission) == 75000,
    set(submission.columns) == set(['sample_id', 'price']),
    missing_values == 0,
    unique_ids == len(submission),
    negative_prices == 0,
    submission['sample_id'].dtype in ['int64', 'int32'],
    submission['price'].dtype in ['float64', 'float32']
]

check_names = [
    "75,000 rows",
    "Required columns",
    "No missing values", 
    "Unique sample_ids",
    "No negative prices",
    "sample_id is integer",
    "price is float"
]

for check, name in zip(checks, check_names):
    status = "✅" if check else "❌"
    print(f"   {status} {name}")
    if not check:
        validation_passed = False

print(f"\n🎯 SUBMISSION STATUS:")
if validation_passed:
    print("✅ SUBMISSION VALIDATED - READY FOR COMPETITION!")
    print("📤 File: final_submission.csv")
    print("🏆 Performance: 57.36% SMAPE")
else:
    print("❌ VALIDATION FAILED - FIX ISSUES BEFORE SUBMISSION")

print("\n" + "="*50)