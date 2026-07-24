import os
import glob
import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import xgboost as xgb
import lime.lime_tabular
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

  
#CONFIGURATION
  
DATA_DIR = 'data/'
OUTPUT_DIR = 'model_backend/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

#Define the features available at each checkpoint
CHECKPOINTS = {
    'CP1': ['total_clicks', 'Online_C', 'Online_O'],
    'CP2': ['total_clicks', 'Online_C', 'Online_O', 'CW1'],
    'CP3': ['total_clicks', 'Online_C', 'Online_O', 'CW1', 'CW2']
}

  
#STEP 1: LOAD ACTIVITY (PRE-PROCESSED MOODLE) FILES
  
print("\n--- STEP 1: Loading Activity (Pre-processed Moodle) Files ---")
act_paths = sorted(glob.glob(os.path.join(DATA_DIR, "Activity*.csv")) + glob.glob(os.path.join(DATA_DIR, "activity*.csv")))

act_frames = []
for p in act_paths:
    try:
        df = pd.read_csv(p, encoding='utf-8', on_bad_lines='skip')
        df.columns = df.columns.str.strip()
        
        for alias in ['RollNumber', 'roll_number', 'StudentID']:
            if alias in df.columns: df.rename(columns={alias: 'RollNumber'}, inplace=True)
        for alias in ['Online C', 'Online_C']:
            if alias in df.columns: df.rename(columns={alias: 'Online_C'}, inplace=True)
        for alias in ['Online O', 'Online_O']:
            if alias in df.columns: df.rename(columns={alias: 'Online_O'}, inplace=True)
        
        if 'RollNumber' in df.columns:
            df.dropna(subset=['RollNumber'], inplace=True)
            df['RollNumber'] = df['RollNumber'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip().str.upper()
            
            if 'Online_C' in df.columns and 'Online_O' in df.columns:
                df['total_clicks'] = df['Online_C'] + df['Online_O']
                act_frames.append(df[['RollNumber', 'Online_C', 'Online_O', 'total_clicks']])
    except Exception as e:
        print(f"  ⚠ Skipped {os.path.basename(p)}: {e}")

if not act_frames:
    print("  [FAIL] CRITICAL: No Activity files loaded. Check data folder.")
    raise SystemExit(1)

activity_data = pd.concat(act_frames, ignore_index=True)
activity_data = activity_data.sort_values('total_clicks', ascending=False).drop_duplicates('RollNumber')
print(f"  [OK] Loaded {len(activity_data)} unique student activity records.")

  
#STEP 2: LOAD MARKS/RESULT FILES
  
print("\n--- STEP 2: Loading Marks/Result Files ---")
res_paths = sorted(glob.glob(os.path.join(DATA_DIR, "Result*.csv")) + glob.glob(os.path.join(DATA_DIR, "result*.csv")))

res_frames = []
for p in res_paths:
    try:
        df = pd.read_csv(p, encoding='utf-8', on_bad_lines='skip')
        df.columns = df.columns.str.strip()
        
        for alias in ['StudentID', 'student_id', 'ID', 'id', 'RollNumber']:
            if alias in df.columns and 'RollNumber' not in df.columns:
                df.rename(columns={alias: 'RollNumber'}, inplace=True)
                
        if 'RollNumber' in df.columns:
            df.dropna(subset=['RollNumber'], inplace=True)
            df['RollNumber'] = df['RollNumber'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip().str.upper()
            res_frames.append(df)
    except Exception as e:
        print(f"  ! Skipped {os.path.basename(p)}: {e}")

marks_data = pd.concat(res_frames, ignore_index=True)
marks_data.drop_duplicates('RollNumber', keep='first', inplace=True)
print(f"  [OK] Loaded {len(marks_data)} unique student mark records.")

  
#STEP 3: MERGING MASTER DATASET & DERIVING TARGET
  
print("\n--- STEP 3: Merging to Create Master Dataset ---")
df = pd.merge(activity_data, marks_data, on='RollNumber', how='inner')
df.rename(columns={'RollNumber': 'StudentID'}, inplace=True)

target_col = None
for alias in ['ESE', 'EndSemesterExam', 'Total']:
    if alias in df.columns:
        target_col = alias
        break

if target_col:
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df.dropna(subset=[target_col], inplace=True)
    df['target'] = (df[target_col] < 60).astype(int)
    print(f"  [OK] Derived 'target' from {target_col} (< 60 is At-Risk).")
else:
    print("  [FAIL] CRITICAL: Could not find ESE column to derive target.")
    raise SystemExit(1)

for col in ['total_clicks', 'Online_C', 'Online_O', 'CW1', 'CW2']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

print(f"  [OK] Master Dataset Built: {len(df)} total students ready for training.")

print(f"\n--- DIAGNOSTICS: Full Dataset ---")
print(f"Class distribution in FULL dataset:")
print(df['target'].value_counts(normalize=True))
print(f"\nFeature statistics:")
print(df[['total_clicks', 'Online_C', 'Online_O', 'CW1', 'CW2']].describe())

# *** CORRELATION ANALYSIS ***
print(f"\n--- CORRELATION ANALYSIS ---")
print(f"Pearson correlation between features and target (ESE < 60):")
correlation = df[['total_clicks', 'Online_C', 'Online_O', 'CW1', 'CW2', target_col, 'target']].corr()
print(correlation[['target']].sort_values('target', ascending=False))
print(f"\nInterpretation: Values > 0.3 indicate moderate correlation")

  
# STEP 4: MODEL TRAINING & RIGOROUS LIME EVALUATION
  
print("\n--- STEP 4: Training XGBoost Models & Evaluating Explanations ---")

trained_models = {}

  
# VARIANT: REGRESSION MODEL (predict actual ESE score)
print("\n--- REGRESSION VARIANT: Predicting Actual ESE Scores ---")
regression_models = {}

for cp_name, features in CHECKPOINTS.items():
    print(f"\n> Regression {cp_name} ({len(features)} features)")
    
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"  ! Skipping {cp_name} due to missing features: {missing_features}")
        continue
    
    X = df[features]
    y_actual_ese = df[target_col]  # Use actual ESE scores, not binary
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_actual_ese, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.01, random_state=42)
    model.fit(X_train, y_train)    
    
    #Predict ESE scores
    y_pred_scores = model.predict(X_test)
    
    #Convert predictions to binary (< 60 = at-risk)
    y_pred_binary = (y_pred_scores < 60).astype(int)
    y_test_binary = (y_test < 60).astype(int)
    
    #Calculate metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    acc = accuracy_score(y_test_binary, y_pred_binary)
    rec = recall_score(y_test_binary, y_pred_binary)
    f1 = f1_score(y_test_binary, y_pred_binary)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_scores))
    mae = mean_absolute_error(y_test, y_pred_scores)
    r2 = r2_score(y_test, y_pred_scores)
    
    print(f"  [REGRESSION] Acc {acc:.2f} | Recall {rec:.2f} | F1 {f1:.2f}")
    print(f"  [ESE PREDICTION] RMSE {rmse:.2f} | MAE {mae:.2f} | R² {r2:.3f}")
    
    # LIME evaluation; regression model
    my_kernel_width = 0.4 * np.sqrt(len(features))
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        training_labels=(y_train < 60).astype(int).values,
        feature_names=features,
        class_names=['Safe', 'At-Risk'],
        mode='classification',
        kernel_width=my_kernel_width,
        random_state=42
    )
    
    eval_sample_size = min(50, len(X_test))
    sample_idx = np.random.choice(len(X_test), eval_sample_size, replace=False)
    X_sample = X_test.iloc[sample_idx]
    
    fidelities = []
    for i in range(len(X_sample)):
        inst = X_sample.iloc[i].values
        pred_score = model.predict([inst])[0]
        pred_binary = 1 if pred_score < 60 else 0
        
        exp = explainer.explain_instance(inst, lambda x: np.column_stack([np.ones(len(x)), np.ones(len(x))]), num_features=len(features), num_samples=3000)
        
        class_match = 1 if pred_binary == exp.predict_proba.argmax() else 0
        fidelities.append(class_match)
    
    avg_fidelity = np.mean(fidelities)
    
    print(f"  [LIME] Class Fidelity {avg_fidelity:.2%}")
    
    regression_models[cp_name] = {
        'model': model,
        'explainer_data': X_train.values,
        'metrics': {
            'accuracy': acc,
            'recall': rec,
            'f1': f1,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'fidelity': avg_fidelity
        }
    }

for cp_name, features in CHECKPOINTS.items():
    print(f"\n> Processing {cp_name} ({len(features)} features)")
    
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"  ! Skipping {cp_name} due to missing features: {missing_features}")
        continue
        
    X = df[features]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"  Train/Test split for {cp_name}:")
    print(f"    Training set class distribution (before SMOTE):")
    print(f"      {y_train.value_counts(normalize=True).to_dict()}")
    print(f"    Test set class distribution:")
    print(f"      {y_test.value_counts(normalize=True).to_dict()}")

    #Apply SMOTE only to the training data
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    #Train model on the balanced SMOTE data
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train_res, y_train_res)
    
    #CUSTOM 45% THRESHOLD
    custom_threshold = 0.45
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= custom_threshold).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    my_kernel_width = 0.4 * np.sqrt(len(features))
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_res.values,     
        training_labels=y_train_res.values,    
        feature_names=features,
        class_names=['Safe', 'At-Risk'],
        mode='classification',
        discretize_continuous=True,
        discretizer='entropy',
        kernel_width=my_kernel_width,
        random_state=42
    )
    
    # --- DYNAMIC FIDELITY EVALUATION ---
    eval_sample_size = min(50, len(X_test))
    sample_idx = np.random.choice(len(X_test), eval_sample_size, replace=False)
    X_sample = X_test.iloc[sample_idx]
    
    fidelities = []
    r2_scores = []
    
    for i in range(len(X_sample)):
        inst = X_sample.iloc[i].values
        xgb_prob = model.predict_proba([inst])[0][1]
        
        exp = explainer.explain_instance(inst, model.predict_proba, num_features=len(features), num_samples=3000)
        
        r2_scores.append(exp.score)
        
        if exp.local_pred.shape[0] == 2:
            lime_prob = exp.local_pred[1]
        else:
            lime_prob = exp.local_pred[0]
            
        class_match = 1 if (xgb_prob >= 0.45) == (lime_prob >= 0.45) else 0
        fidelities.append(class_match)
        
    avg_fidelity = np.mean(fidelities)
    avg_r2 = np.mean(r2_scores)
    
    # --- DYNAMIC STABILITY EVALUATION (Cosine Similarity) ---
    probs = model.predict_proba(X_test)[:, 1]
    risk_indices = np.where(probs > 0.5)[0]
    
    if len(risk_indices) > 0:
        target_inst = X_test.iloc[risk_indices[0]].values
    else:
        target_inst = X_test.iloc[0].values
        
    n_runs = 30
    weight_vectors = []
    
    for _ in range(n_runs):
        exp = explainer.explain_instance(target_inst, model.predict_proba, num_features=len(features), num_samples=3000)
        sorted_weights = sorted(exp.local_exp[1], key=lambda x: x[0])
        vector = np.array([w for _, w in sorted_weights])
        weight_vectors.append(vector)
        
    sim_matrix = cosine_similarity(np.array(weight_vectors))
    upper_tri = np.triu_indices(n_runs, k=1)
    stability_score = np.mean(sim_matrix[upper_tri])
    
    print(f"  [OK] Metrics: Acc {acc:.2f} | Recall {rec:.2f} | F1 {f1:.2f}")
    print(f"  [OK] LIME Evaluation: Class Fidelity {avg_fidelity:.2%} | Local R^2 {avg_r2:.3f} | Stability {stability_score:.2%}")
    
    trained_models[cp_name] = {
        'model': model,
        'explainer_data': X_train_res.values, 
        'metrics': {
            'accuracy': acc,
            'recall': rec,
            'f1': f1,
            'fidelity': avg_fidelity,
            'r2_score': avg_r2,
            'stability': stability_score
        }
    }

  
#STEP 5: EXPORT
  
print("\n--- STEP 5: Exporting Models for Analysis & Dashboard ---")

if not trained_models:
    print("  [FAIL] CRITICAL: No models were successfully trained.")
    raise SystemExit(1)

joblib.dump(trained_models, os.path.join(OUTPUT_DIR, 'checkpoint_models.pkl'))
joblib.dump(regression_models, os.path.join(OUTPUT_DIR, 'checkpoint_models_regression.pkl'))

# Save comparison report
with open(os.path.join(OUTPUT_DIR, 'model_comparison.txt'), 'w') as f:
    f.write("CLASSIFICATION vs REGRESSION COMPARISON\n")
    f.write("="*60 + "\n\n")
    f.write("CLASSIFICATION (Binary: ESE < 60 = At-Risk)\n")
    f.write("="*60 + "\n")
    for cp, data in trained_models.items():
        f.write(f"\n{cp}:\n")
        for metric, value in data['metrics'].items():
            f.write(f"  {metric}: {value}\n")
    f.write("\n\nREGRESSION (Predict Actual ESE Score)\n")
    f.write("="*60 + "\n")
    for cp, data in regression_models.items():
        f.write(f"\n{cp}:\n")
        for metric, value in data['metrics'].items():
            f.write(f"  {metric}: {value}\n")
            
export_cols = ['StudentID', 'total_clicks', 'Online_C', 'Online_O', 'CW1', 'CW2', 'target']
export_cols = [c for c in export_cols if c in df.columns]
df[export_cols].to_csv(os.path.join(OUTPUT_DIR, 'processed_students.csv'), index=False)

print(f"  [OK] {OUTPUT_DIR}checkpoint_models.pkl saved successfully.")
print(f"  [OK] {OUTPUT_DIR}processed_students.csv saved successfully.")
print("\n================================================================")
print("PIPELINE COMPLETE.")
print("================================================================")