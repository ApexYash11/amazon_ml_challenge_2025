# 🎯 ULTRA-ADVANCED PIPELINE FOR 43% SMAPE TARGET
# Deep feature engineering + advanced techniques

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, f_regression
import re
import warnings
warnings.filterwarnings('ignore')

print("🎯 ULTRA-ADVANCED PIPELINE FOR 43% SMAPE TARGET")
print("Deep feature engineering + stacking")
print("="*60)

# =============================================================================
# ULTRA-DEEP FEATURE ENGINEERING
# =============================================================================

def create_ultra_features(df):
    """Create ultra-deep price-predictive features"""
    
    print("   🔬 Creating ultra-deep features...")
    
    features = pd.DataFrame()
    text = df['catalog_content'].fillna('').str.lower()
    
    # === BASIC FEATURES ===
    features['text_length'] = df['catalog_content'].str.len().fillna(0)
    features['word_count'] = text.str.split().str.len().fillna(0)
    features['avg_word_length'] = features['text_length'] / (features['word_count'] + 1)
    features['sentence_count'] = df['catalog_content'].str.count(r'[.!?]').fillna(0)
    
    # === PRICE INDICATOR FEATURES ===
    
    # Luxury/Premium indicators
    luxury_terms = ['premium', 'luxury', 'professional', 'deluxe', 'superior', 'high-end', 
                   'top-quality', 'exclusive', 'elite', 'platinum', 'gold', 'diamond']
    features['luxury_score'] = sum([text.str.contains(term, regex=False).astype(int) for term in luxury_terms])
    
    # Budget indicators
    budget_terms = ['cheap', 'budget', 'affordable', 'economy', 'basic', 'simple', 
                   'low-cost', 'discount', 'clearance', 'sale']
    features['budget_score'] = sum([text.str.contains(term, regex=False).astype(int) for term in budget_terms])
    
    # Brand strength indicators
    brand_terms = ['brand', 'branded', 'original', 'authentic', 'genuine', 'official',
                  'certified', 'licensed', 'authorized']
    features['brand_strength'] = sum([text.str.contains(term, regex=False).astype(int) for term in brand_terms])
    
    # === MATERIAL VALUE FEATURES ===
    
    # High-value materials
    expensive_materials = ['steel', 'stainless', 'aluminum', 'titanium', 'carbon', 'leather',
                          'silk', 'wool', 'cashmere', 'gold', 'silver', 'platinum', 'crystal']
    features['expensive_material_score'] = sum([text.str.contains(mat, regex=False).astype(int) for mat in expensive_materials])
    
    # Low-value materials
    cheap_materials = ['plastic', 'synthetic', 'artificial', 'fabric', 'cloth', 'paper',
                      'cardboard', 'foam', 'rubber']
    features['cheap_material_score'] = sum([text.str.contains(mat, regex=False).astype(int) for mat in cheap_materials])
    
    # === CATEGORY-SPECIFIC FEATURES ===
    
    # Electronics (typically higher value)
    electronics = ['phone', 'smartphone', 'laptop', 'computer', 'tablet', 'monitor', 
                  'camera', 'headphone', 'speaker', 'smartwatch', 'drone', 'console']
    features['electronics_score'] = sum([text.str.contains(item, regex=False).astype(int) for item in electronics])
    
    # Jewelry/Accessories (high value)
    jewelry = ['jewelry', 'jewellery', 'watch', 'ring', 'necklace', 'bracelet', 
              'earring', 'pendant', 'chain', 'charm']
    features['jewelry_score'] = sum([text.str.contains(item, regex=False).astype(int) for item in jewelry])
    
    # Home/Furniture (medium-high value)
    furniture = ['furniture', 'chair', 'table', 'desk', 'bed', 'sofa', 'couch',
                'cabinet', 'shelf', 'dresser', 'wardrobe']
    features['furniture_score'] = sum([text.str.contains(item, regex=False).astype(int) for item in furniture])
    
    # Clothing (variable value)
    clothing = ['shirt', 'dress', 'pants', 'jeans', 'jacket', 'coat', 'sweater',
               'shoes', 'boots', 'sneakers', 'sandals']
    features['clothing_score'] = sum([text.str.contains(item, regex=False).astype(int) for item in clothing])
    
    # === SIZE/QUANTITY FEATURES ===
    
    # Size specifications (often correlate with price)
    features['has_measurements'] = text.str.contains(r'\d+\s*(inch|cm|mm|ft|meter)', regex=True).astype(int)
    features['has_weight'] = text.str.contains(r'\d+\s*(kg|gram|lb|oz)', regex=True).astype(int)
    features['has_volume'] = text.str.contains(r'\d+\s*(liter|ml|gallon)', regex=True).astype(int)
    
    # Pack/set indicators
    features['is_set'] = text.str.contains(r'\b(set|pack|bundle|kit)\b', regex=True).astype(int)
    features['quantity_mentioned'] = text.str.contains(r'\d+\s*(piece|pcs|items)', regex=True).astype(int)
    
    # === TECHNOLOGY FEATURES ===
    
    # Modern tech features
    tech_features = ['smart', 'digital', 'electronic', 'wireless', 'bluetooth', 'wifi',
                    'usb', 'led', 'lcd', 'oled', 'hd', '4k', 'ai', 'app']
    features['tech_score'] = sum([text.str.contains(feat, regex=False).astype(int) for feat in tech_features])
    
    # === CONDITION FEATURES ===
    
    # New condition indicators
    new_indicators = ['new', 'brand new', 'fresh', 'latest', 'current', 'modern', 'recent']
    features['new_condition_score'] = sum([text.str.contains(ind, regex=False).astype(int) for ind in new_indicators])
    
    # Used condition indicators
    used_indicators = ['used', 'second hand', 'pre-owned', 'refurbished', 'vintage', 'antique']
    features['used_condition_score'] = sum([text.str.contains(ind, regex=False).astype(int) for ind in used_indicators])
    
    # === COMPLEXITY FEATURES ===
    
    # Product complexity (complex products often cost more)
    features['complex_words'] = text.str.findall(r'\b\w{10,}\b').str.len().fillna(0)
    features['technical_terms'] = text.str.contains(r'specification|feature|technology|advanced|professional', regex=True).astype(int)
    features['caps_ratio'] = df['catalog_content'].str.count(r'[A-Z]').fillna(0) / (features['text_length'] + 1)
    
    # === NUMERIC FEATURES ===
    
    # Extract all numbers from text
    features['number_count'] = text.str.findall(r'\d+').str.len().fillna(0)
    features['large_numbers'] = text.str.findall(r'\d{3,}').str.len().fillna(0)
    features['decimal_numbers'] = text.str.contains(r'\d+\.\d+', regex=True).astype(int)
    
    # Model numbers (often indicate premium products)
    features['has_model_number'] = text.str.contains(r'\b[a-z]+\d+[a-z]*\b', regex=True).astype(int)
    
    # === IMAGE FEATURES ===
    
    features['image_link_length'] = df['image_link'].str.len().fillna(0)
    features['has_https'] = df['image_link'].str.contains('https://', na=False).astype(int)
    features['image_format_jpg'] = df['image_link'].str.contains('.jpg', na=False).astype(int)
    features['image_format_png'] = df['image_link'].str.contains('.png', na=False).astype(int)
    features['multiple_images'] = df['image_link'].str.count(',').fillna(0)
    
    # === INTERACTION FEATURES ===
    
    # Create meaningful interactions
    features['luxury_tech_combo'] = features['luxury_score'] * features['tech_score']
    features['brand_electronics_combo'] = features['brand_strength'] * features['electronics_score']
    features['material_quality_ratio'] = features['expensive_material_score'] / (features['cheap_material_score'] + 1)
    features['condition_premium_combo'] = features['new_condition_score'] * features['luxury_score']
    
    # Text density features
    features['info_density'] = (features['number_count'] + features['tech_score'] + features['luxury_score']) / (features['word_count'] + 1)
    
    print(f"   ✅ Created {features.shape[1]} ultra-deep features")
    
    return features

# =============================================================================
# ADVANCED TEXT VECTORIZATION
# =============================================================================

def create_advanced_text_features(train_text, test_text):
    """Create multiple types of text features"""
    
    print("   📝 Creating advanced text vectors...")
    
    all_train_features = []
    all_test_features = []
    
    # 1. Character-level TF-IDF
    print("     🔤 Character-level TF-IDF...")
    char_vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),
        max_features=200,
        lowercase=True
    )
    train_char = char_vectorizer.fit_transform(train_text)
    test_char = char_vectorizer.transform(test_text)
    all_train_features.append(train_char.toarray())
    all_test_features.append(test_char.toarray())
    
    # 2. Word-level TF-IDF (multiple configs)
    tfidf_configs = [
        {'max_features': 300, 'ngram_range': (1, 1), 'min_df': 3},
        {'max_features': 200, 'ngram_range': (1, 2), 'min_df': 2},
        {'max_features': 100, 'ngram_range': (2, 3), 'min_df': 2},
    ]
    
    for i, config in enumerate(tfidf_configs):
        print(f"     📚 Word TF-IDF {i+1}...")
        vectorizer = TfidfVectorizer(
            max_features=config['max_features'],
            ngram_range=config['ngram_range'],
            min_df=config['min_df'],
            stop_words='english',
            lowercase=True,
            token_pattern=r'\b[a-zA-Z]{2,}\b'
        )
        
        train_tfidf = vectorizer.fit_transform(train_text)
        test_tfidf = vectorizer.transform(test_text)
        all_train_features.append(train_tfidf.toarray())
        all_test_features.append(test_tfidf.toarray())
    
    # 3. Count Vectorizer for frequency features
    print("     📊 Count vectorizer...")
    count_vectorizer = CountVectorizer(
        max_features=150,
        ngram_range=(1, 2),
        min_df=3,
        stop_words='english'
    )
    train_counts = count_vectorizer.fit_transform(train_text)
    test_counts = count_vectorizer.transform(test_text)
    all_train_features.append(train_counts.toarray())
    all_test_features.append(test_counts.toarray())
    
    # Combine all text features
    train_combined = np.hstack(all_train_features)
    test_combined = np.hstack(all_test_features)
    
    print(f"   ✅ Total text features: {train_combined.shape[1]}")
    
    return train_combined, test_combined

# =============================================================================
# STACKED ENSEMBLE WITH MULTIPLE LEVELS
# =============================================================================

def create_stacked_ensemble(X_train, y_train, X_val, y_val):
    """Create advanced stacked ensemble"""
    
    print("   🏗️ Building stacked ensemble...")
    
    # Level 1 models (base models)
    level1_models = {
        'xgb1': xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, verbosity=0),
        'xgb2': xgb.XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.03, random_state=43, verbosity=0),
        'lgb1': lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, verbose=-1),
        'lgb2': lgb.LGBMRegressor(n_estimators=300, max_depth=8, learning_rate=0.03, random_state=43, verbose=-1),
        'rf': RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1),
        'et': ExtraTreesRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1),
    }
    
    # Train level 1 models and get predictions
    level1_predictions = np.zeros((len(X_val), len(level1_models)))
    level1_test_predictions = []
    
    print("     🤖 Training level 1 models...")
    for i, (name, model) in enumerate(level1_models.items()):
        print(f"       Training {name}...")
        model.fit(X_train, y_train)
        level1_predictions[:, i] = model.predict(X_val)
        level1_test_predictions.append(model)
    
    # Level 2 model (meta-learner)
    print("     🧠 Training meta-learner...")
    
    # Scale level 1 predictions
    scaler = StandardScaler()
    level1_scaled = scaler.fit_transform(level1_predictions)
    
    # Use Ridge as meta-learner
    meta_model = Ridge(alpha=0.1)
    meta_model.fit(level1_scaled, y_val)
    
    # Calculate stacked performance
    stacked_pred = meta_model.predict(level1_scaled)
    
    def calculate_smape(y_true, y_pred):
        # Inverse transform
        y_true_orig = np.expm1(y_true) - 0.01
        y_pred_orig = np.expm1(y_pred) - 0.01
        y_pred_orig = np.maximum(y_pred_orig, 0.01)
        return 100 * np.mean(2 * np.abs(y_pred_orig - y_true_orig) / (np.abs(y_true_orig) + np.abs(y_pred_orig)))
    
    stacked_smape = calculate_smape(y_val, stacked_pred)
    
    print(f"     ✅ Stacked ensemble SMAPE: {stacked_smape:.2f}%")
    
    return level1_models, meta_model, scaler, stacked_smape

# =============================================================================
# MAIN ULTRA PIPELINE
# =============================================================================

def run_ultra_pipeline():
    """Run complete ultra-advanced pipeline"""
    
    start_time = time.time()
    
    try:
        # Load data
        print("📂 Loading data...")
        train = pd.read_csv('dataset/train.csv')
        test = pd.read_csv('dataset/test.csv')
        
        # Create ultra-deep features
        print("\n🔬 Ultra-Deep Feature Engineering...")
        train_features = create_ultra_features(train)
        test_features = create_ultra_features(test)
        
        # Advanced text features
        train_text_features, test_text_features = create_advanced_text_features(
            train['catalog_content'].fillna(''),
            test['catalog_content'].fillna('')
        )
        
        # Combine all features
        X_train = np.hstack([train_features.values, train_text_features])
        X_test = np.hstack([test_features.values, test_text_features])
        
        print(f"   ✅ Total ultra features: {X_train.shape[1]}")
        
        # Target transformation
        print("\n🔄 Target Transformation...")
        y_train = np.log1p(train['price'].values + 0.01)
        
        print(f"   📊 Transformed target skewness: {pd.Series(y_train).skew():.3f}")
        
        # Feature selection to prevent overfitting
        print("\n🎯 Feature Selection...")
        selector = SelectKBest(f_regression, k=min(800, X_train.shape[1]))
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        print(f"   ✅ Selected features: {X_train_selected.shape[1]}")
        
        # Train-validation split
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train_selected, y_train, test_size=0.2, random_state=42
        )
        
        # Create stacked ensemble
        print("\n🏗️ Stacked Ensemble...")
        level1_models, meta_model, scaler, ensemble_smape = create_stacked_ensemble(
            X_train_split, y_train_split, X_val_split, y_val_split
        )
        
        print(f"\n🏆 ULTRA ENSEMBLE PERFORMANCE:")
        print(f"   🎯 Validation SMAPE: {ensemble_smape:.2f}%")
        
        if ensemble_smape < 55:
            print(f"   🎉 EXCELLENT! Major improvement!")
        if ensemble_smape < 50:
            print(f"   🚀 OUTSTANDING! Getting very close to target!")
        if ensemble_smape < 45:
            print(f"   🏆 AMAZING! Competition-winning level!")
        
        # Final predictions
        print(f"\n🔮 Final Predictions...")
        
        # Retrain on full data
        for model in level1_models.values():
            model.fit(X_train_selected, y_train)
        
        # Get level 1 predictions on test set
        test_level1_preds = np.zeros((len(X_test_selected), len(level1_models)))
        for i, model in enumerate(level1_models.values()):
            test_level1_preds[:, i] = model.predict(X_test_selected)
        
        # Scale and get final predictions
        test_level1_scaled = scaler.transform(test_level1_preds)
        final_predictions = meta_model.predict(test_level1_scaled)
        
        # Inverse transform
        final_prices = np.expm1(final_predictions) - 0.01
        final_prices = np.maximum(final_prices, 0.01)
        
        # Create submission
        submission = pd.DataFrame({
            'sample_id': test['sample_id'].astype(int),
            'price': final_prices.astype(float)
        })
        
        filename = f'competition_submissions/ultra_stacked_{ensemble_smape:.1f}_smape.csv'
        submission.to_csv(filename, index=False, float_format='%.4f')
        
        elapsed_time = time.time() - start_time
        
        print(f"\n🎉 ULTRA PIPELINE COMPLETE!")
        print(f"⏱️ Total time: {elapsed_time/60:.1f} minutes")
        print(f"🎯 Validation SMAPE: {ensemble_smape:.2f}%")
        print(f"📤 Submission: {filename}")
        print(f"📊 Price range: ${final_prices.min():.2f} - ${final_prices.max():.2f}")
        
        if ensemble_smape < 50:
            print(f"\n🏆 COMPETITION-LEVEL PERFORMANCE ACHIEVED!")
            print(f"🎯 Ready to challenge the 43% SMAPE leaders!")
        
        return submission, ensemble_smape
        
    except Exception as e:
        print(f"❌ Ultra pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    import time
    submission, smape = run_ultra_pipeline()