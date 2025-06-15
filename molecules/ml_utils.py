import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import os
from datetime import datetime

def prepare_training_data(compounds, target='field'):
    """
    화합물 데이터를 학습용 데이터로 준비합니다.
    """
    # 특성 데이터 준비
    features = []
    targets = []
    
    for compound in compounds:
        # 타겟 값 확인
        if target == 'field' and compound.field_halflife is None:
            continue
        elif target == 'lab' and compound.lab_halflife is None:
            continue
        
        # 특성 딕셔너리 생성
        feature_dict = {
            'molecular_weight': compound.molecular_weight or 0,
            'logp': compound.logp or 0,
            'tpsa': compound.tpsa or 0,
            'num_h_donors': compound.num_h_donors or 0,
            'num_h_acceptors': compound.num_h_acceptors or 0,
            'num_rotatable_bonds': compound.num_rotatable_bonds or 0,
            'num_aromatic_rings': compound.num_aromatic_rings or 0,
            'num_heavy_atoms': compound.num_heavy_atoms or 0,
            'active_ingredient_content': compound.active_ingredient_content or 0,
        }
        
        # 계통 원핫인코딩
        systems = ['Organophosphate', 'Triazole', 'Carbamate', 'Amide', 
                  'Sulfonylurea', 'Anilide', 'Strobilurin', 'Pyrazole']
        for sys in systems:
            feature_dict[f'system_{sys}'] = 1 if compound.system == sys else 0
        
        features.append(feature_dict)
        
        # 타겟 값
        if target == 'field':
            targets.append(compound.field_halflife)
        else:
            targets.append(compound.lab_halflife)
    
    # DataFrame으로 변환
    X = pd.DataFrame(features)
    y = np.array(targets)
    
    # NaN 값을 0으로 대체
    X = X.fillna(0)
    
    return X, y, list(X.columns)

def train_models(X, y, target='field', test_size=0.2, random_state=42):
    """
    여러 모델을 학습하고 성능을 평가합니다.
    """
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 모델 정의
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Random Forest': RandomForestRegressor(
            n_estimators=100, 
            random_state=random_state,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100,
            random_state=random_state
        ),
        'SVM Regression': SVR(kernel='rbf', C=1.0)
    }
    
    results = []
    
    for name, model in models.items():
        # 모델 학습
        if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'SVM Regression']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # 성능 평가
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        # 교차 검증
        if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'SVM Regression']:
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        else:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        # 특성 중요도 (가능한 경우)
        feature_importances = {}
        if hasattr(model, 'feature_importances_'):
            for i, importance in enumerate(model.feature_importances_):
                feature_importances[X.columns[i]] = float(importance)
        elif hasattr(model, 'coef_'):
            for i, coef in enumerate(model.coef_):
                feature_importances[X.columns[i]] = float(abs(coef))
        
        result = {
            'name': name,
            'model': model,
            'scaler': scaler if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'SVM Regression'] else None,
            'r2_score': r2,
            'rmse': rmse,
            'mae': mae,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importances': feature_importances,
            'target': target,
            'training_samples': len(X_train),
            'feature_names': list(X.columns)
        }
        
        results.append(result)
    
    # R2 점수로 정렬
    results.sort(key=lambda x: x['r2_score'], reverse=True)
    
    return results

def save_model(model_info, model_dir='media/models'):
    """
    모델을 파일로 저장합니다.
    """
    os.makedirs(model_dir, exist_ok=True)
    
    # 고유한 파일명 생성
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{model_info['name'].replace(' ', '_')}_{model_info['target']}_{timestamp}.pkl"
    filepath = os.path.join(model_dir, filename)
    
    # 모델과 스케일러 저장
    save_data = {
        'model': model_info['model'],
        'scaler': model_info['scaler'],
        'feature_names': model_info['feature_names'],
        'model_type': model_info['name'],
        'target': model_info['target']
    }
    
    joblib.dump(save_data, filepath)
    
    return filepath

def load_model(filepath):
    """
    저장된 모델을 로드합니다.
    """
    return joblib.load(filepath)

def predict_halflife(model_data, compound):
    """
    단일 화합물의 반감기를 예측합니다.
    """
    # 특성 준비
    feature_dict = {
        'molecular_weight': compound.molecular_weight or 0,
        'logp': compound.logp or 0,
        'tpsa': compound.tpsa or 0,
        'num_h_donors': compound.num_h_donors or 0,
        'num_h_acceptors': compound.num_h_acceptors or 0,
        'num_rotatable_bonds': compound.num_rotatable_bonds or 0,
        'num_aromatic_rings': compound.num_aromatic_rings or 0,
        'num_heavy_atoms': compound.num_heavy_atoms or 0,
        'active_ingredient_content': compound.active_ingredient_content or 0,
    }
    
    # 계통 원핫인코딩
    systems = ['Organophosphate', 'Triazole', 'Carbamate', 'Amide', 
              'Sulfonylurea', 'Anilide', 'Strobilurin', 'Pyrazole']
    for sys in systems:
        feature_dict[f'system_{sys}'] = 1 if compound.system == sys else 0
    
    # DataFrame으로 변환
    X = pd.DataFrame([feature_dict])[model_data['feature_names']]
    
    # 예측
    if model_data['scaler']:
        X_scaled = model_data['scaler'].transform(X)
        prediction = model_data['model'].predict(X_scaled)[0]
    else:
        prediction = model_data['model'].predict(X)[0]
    
    # 신뢰구간 계산 (간단한 방법)
    # 실제로는 더 정교한 방법을 사용해야 함
    confidence_interval = prediction * 0.1  # 10% 범위
    
    return {
        'predicted_value': prediction,
        'confidence_lower': max(0, prediction - confidence_interval),
        'confidence_upper': prediction + confidence_interval,
        'input_features': feature_dict
    }