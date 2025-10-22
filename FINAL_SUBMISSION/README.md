# 🏆 Amazon ML Challenge 2025 - Final Submission
## Smart Product Pricing Challenge

### 📤 **SUBMISSION FILES**
- **`final_submission.csv`** - Competition submission file (57.36% SMAPE)
- **`model_pipeline.py`** - Complete model training pipeline
- **`README.md`** - This documentation

---

## 🎯 **FINAL PERFORMANCE**

| Metric | Value | Note |
|--------|-------|------|
| **Validation SMAPE** | **57.36%** | Best achieved performance |
| **Improvement** | **10.5%** | vs 64.12% baseline |
| **Training Time** | **42.2 minutes** | Full pipeline execution |
| **Features** | **800** | Selected from 988 engineered |
| **Models** | **6** | Stacked ensemble architecture |

---

## 🏗️ **MODEL ARCHITECTURE**

### **🔧 Ultra-Advanced Stacked Ensemble**

#### **Level 1: Base Models (6 models)**
```
├── XGBoost Model 1 (n_estimators=200, max_depth=6, lr=0.05)
├── XGBoost Model 2 (n_estimators=300, max_depth=8, lr=0.03)
├── LightGBM Model 1 (n_estimators=200, max_depth=6, lr=0.05)  
├── LightGBM Model 2 (n_estimators=300, max_depth=8, lr=0.03)
├── Random Forest (n_estimators=150, max_depth=12)
└── Extra Trees (n_estimators=150, max_depth=12)
```

#### **Level 2: Meta-Learner**
```
Ridge Regression (alpha=0.1)
├── Input: Level 1 predictions (6 features)
├── Scaling: StandardScaler normalization
└── Output: Final price predictions
```

---

## 🔬 **FEATURE ENGINEERING**

### **📊 Feature Categories (988 → 800 selected)**

#### **1. Ultra-Deep Price Features (38 features)**
- **Luxury Indicators:** premium, luxury, professional, deluxe, superior
- **Budget Indicators:** cheap, budget, affordable, economy, basic
- **Brand Strength:** brand, branded, original, authentic, genuine
- **Material Value:** expensive vs cheap materials classification
- **Category Scoring:** electronics, jewelry, furniture, clothing scores
- **Technology Features:** smart, digital, wireless, bluetooth indicators
- **Condition Analysis:** new vs used condition scoring
- **Complexity Metrics:** technical terms, model numbers, specifications

#### **2. Advanced Text Vectors (950 features)**
- **Character TF-IDF:** 3-5 character ngrams (200 features)
- **Word TF-IDF:** Multiple configurations (600 features)
  - Unigrams (300 features, min_df=3)
  - Bigrams (200 features, min_df=2) 
  - Trigrams (100 features, min_df=2)
- **Count Vectorizer:** Frequency analysis (150 features)

#### **3. Key Innovation Features**
```python
# Price prediction insights
luxury_score = sum of luxury terms in text
material_quality_ratio = expensive_materials / (cheap_materials + 1)
luxury_tech_combo = luxury_score × tech_score
info_density = (numbers + tech + luxury) / word_count
```

---

## 🎯 **TARGET TRANSFORMATION**

### **Log Transformation Strategy**
```python
# Transform target for normality
y_transformed = log(price + 0.01)

# Benefits:
# - Reduced skewness: 13.601 → 0.206
# - Better model performance
# - Handles zero prices gracefully

# Inverse transform for predictions
final_prices = exp(predictions) - 0.01
```

---

## 🤖 **TRAINING METHODOLOGY**

### **1. Data Preparation**
```python
# Dataset
Train: 75,000 samples × 988 features
Test:  75,000 samples × 988 features

# Feature Selection
SelectKBest(f_regression, k=800)
# Prevents overfitting while retaining predictive power
```

### **2. Stacked Training Process**
```python
# Level 1: Train 6 diverse base models
for model in [xgb1, xgb2, lgb1, lgb2, rf, et]:
    model.fit(X_train, y_train)
    level1_predictions.append(model.predict(X_val))

# Level 2: Train meta-learner on Level 1 outputs
meta_model = Ridge(alpha=0.1)
meta_model.fit(scaled_level1_predictions, y_val)
```

### **3. Validation Strategy**
```python
# 80/20 train-validation split
# Cross-validation within each base model
# Final ensemble validation on holdout set
```

---

## 📈 **PERFORMANCE ANALYSIS**

### **Individual Model Performance**
| Model | SMAPE | Contribution |
|-------|-------|-------------|
| XGBoost 1 | ~58.5% | High weight |
| XGBoost 2 | ~58.8% | High weight |
| LightGBM 1 | ~59.2% | Medium weight |
| LightGBM 2 | ~59.5% | Medium weight |
| Random Forest | ~60.1% | Low weight |
| Extra Trees | ~60.3% | Low weight |

### **Ensemble Benefit**
- **Best Individual:** ~58.5% SMAPE
- **Stacked Ensemble:** **57.36% SMAPE**
- **Improvement:** 1.14 percentage points from stacking

---

## 🚀 **EXECUTION INSTRUCTIONS**

### **Requirements**
```bash
# Core ML libraries
pip install pandas numpy scikit-learn
pip install xgboost lightgbm
pip install matplotlib seaborn

# Ensure dataset structure:
dataset/
├── train.csv
└── test.csv
```

### **Run Complete Pipeline**
```bash
# Execute full training and prediction pipeline
python model_pipeline.py

# Output:
# - Trains stacked ensemble
# - Generates predictions
# - Creates submission file
# - Reports performance metrics
```

### **Quick Validation**
```python
# Load and validate submission
import pandas as pd
submission = pd.read_csv('final_submission.csv')

print(f"Samples: {len(submission)}")
print(f"Columns: {submission.columns.tolist()}")
print(f"Price range: ${submission['price'].min():.2f} - ${submission['price'].max():.2f}")
print(f"Mean price: ${submission['price'].mean():.2f}")
```

---

## 🔍 **SUBMISSION VALIDATION**

### **Format Verification**
```python
# Required format
final_submission.csv:
├── sample_id: int (1, 3, 9, 19, 20, ...)
├── price: float (predictions in dollars)
├── 75,000 rows
└── No missing values
```

### **Quality Checks**
- ✅ **No negative prices:** All prices ≥ $0.01
- ✅ **Reasonable range:** $0.33 - $305.65
- ✅ **Complete predictions:** 75,000 samples
- ✅ **Proper format:** CSV with required columns
- ✅ **Data types:** sample_id (int), price (float)

---

## 🎯 **COMPETITION STRATEGY**

### **Why This Approach Works**
1. **Domain-Specific Features:** Price indicators beat general embeddings
2. **Ensemble Diversity:** Multiple algorithms capture different patterns  
3. **Stacking Architecture:** Meta-learner optimizes model combinations
4. **Feature Selection:** Prevents overfitting with large feature sets
5. **Target Transformation:** Improves model performance on skewed prices

### **Innovation Highlights**
- **988 engineered features** with deep domain knowledge
- **Character + word level** text analysis
- **6-model stacked ensemble** with Ridge meta-learner
- **Price-specific indicators** (luxury, materials, categories)
- **Robust validation** preventing overfitting

---

## 📊 **RESULTS SUMMARY**

### **Achievement**
- 🎯 **Target:** Competitive Amazon ML Challenge 2025 performance
- ✅ **Achieved:** 57.36% SMAPE (10.5% improvement over baseline)
- 🏆 **Status:** Competition-ready submission

### **Journey**
1. **Baseline XGBoost:** 64.12% SMAPE
2. **Advanced Features:** 60.05% SMAPE  
3. **Ultra Stacked:** **57.36% SMAPE** ⭐

### **Impact**
- **Significant improvement** from systematic feature engineering
- **Robust ensemble** providing stable predictions
- **Production-ready pipeline** for similar price prediction tasks

---

## 👥 **Team Information**

**Team:** Hidden Layers  
**Leader:** Yash Maheshwari  
**Members:** Aaryaman Patti, Raj Badlani  
**Competition:** Amazon ML Challenge 2025 - Smart Product Pricing  

---

## 🏁 **FINAL NOTES**

This submission represents the culmination of extensive experimentation with:
- Traditional ML approaches (XGBoost, LightGBM, RF)
- Deep learning embeddings (BERT + ResNet50) 
- Advanced feature engineering
- Ensemble techniques
- Stacking architectures

**Key Learning:** Domain expertise and careful feature engineering often outperform complex deep learning approaches for structured prediction tasks.

**🚀 Ready for submission to Amazon ML Challenge 2025 leaderboard!**

---

*Generated on October 12, 2025*  
*Best validation SMAPE: 57.36%*  
*Competition-ready submission*