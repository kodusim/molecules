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
import random

# 반감기 임계값 설정 (일)
FIELD_HALFLIFE_THRESHOLD = 30  # 30일 이상이면 지속성 높음(1), 미만이면 낮음(0)
LAB_HALFLIFE_THRESHOLD = 20    # 20일 이상이면 지속성 높음(1), 미만이면 낮음(0)

def prepare_training_data(compounds, target='field'):
    """
    화합물 데이터를 학습용 데이터로 준비합니다.
    """
    # 특성 데이터 준비
    features = []
    targets = []
    
    for compound in compounds:
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
        }
        
        # 계통 원핫인코딩
        systems = ['Organophosphate', 'Triazole', 'Carbamate', 'Amide', 
                  'Sulfonylurea', 'Anilide', 'Strobilurin', 'Pyrazole']
        for sys in systems:
            feature_dict[f'system_{sys}'] = 1 if compound.system == sys else 0
        
        features.append(feature_dict)
        
        # 타겟 값 (없으면 랜덤 생성 - 시연용)
        if target == 'field':
            if compound.field_halflife:
                targets.append(compound.field_halflife)
            else:
                # 시연용 랜덤 값 생성 (10-100일 범위)
                targets.append(random.uniform(10, 100))
        else:
            if compound.lab_halflife:
                targets.append(compound.lab_halflife)
            else:
                # 시연용 랜덤 값 생성 (5-80일 범위)
                targets.append(random.uniform(5, 80))
    
    # DataFrame으로 변환
    X = pd.DataFrame(features)
    y = np.array(targets)
    
    # NaN 값을 0으로 대체
    X = X.fillna(0)
    
    return X, y, list(X.columns)

def train_models(X, y, target='field', test_size=0.2, random_state=42):
    """
    시연용 모델 학습 - 실제로는 학습하지 않고 랜덤 성능 지표 생성
    """
    # 데이터 분할 (형식상)
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
        'Random Forest': RandomForestRegressor(n_estimators=10, random_state=random_state, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=10, random_state=random_state),
        'SVM Regression': SVR(kernel='rbf', C=1.0)
    }
    
    results = []
    
    # 각 모델에 대해 랜덤 성능 지표 생성
    for name, model in models.items():
        # 모델별로 다른 R2 범위 설정
        if name in ['Random Forest', 'Gradient Boosting']:
            # 상위 모델: 0.5 ~ 0.6
            r2 = random.uniform(0.5, 0.6)
        elif name in ['Ridge Regression', 'Lasso Regression']:
            # 중간 모델: 0.4 ~ 0.5
            r2 = random.uniform(0.4, 0.5)
        else:  # Linear Regression, SVM
            # 하위 모델: 0.3 ~ 0.4
            r2 = random.uniform(0.3, 0.4)
        
        # RMSE와 MAE 계산 (R2와 반비례하도록)
        # R2가 낮아졌으므로 RMSE는 더 커짐
        rmse = (1 - r2) * 30 + random.uniform(5, 10)  # 기존 20 -> 30으로 증가
        mae = rmse * 0.8 + random.uniform(-2, 2)
        
        # 교차 검증 점수 (R2 근처 값들로)
        cv_scores = np.array([r2 + random.uniform(-0.05, 0.05) for _ in range(5)])
        cv_scores = np.clip(cv_scores, 0, 1)
        
        # 특성 중요도 (랜덤하게 생성)
        feature_importances = {}
        if name in ['Random Forest', 'Gradient Boosting']:
            importance_values = np.random.random(len(X.columns))
            importance_values = importance_values / importance_values.sum()
            for i, col in enumerate(X.columns):
                feature_importances[col] = float(importance_values[i])
        
        # 이진 분류 성능 지표 추가
        threshold = FIELD_HALFLIFE_THRESHOLD if target == 'field' else LAB_HALFLIFE_THRESHOLD
        
        # 시연용 분류 지표 (R2에 비례하게 조정)
        # R2가 낮아졌으므로 분류 성능도 낮게 설정
        if name in ['Random Forest', 'Gradient Boosting']:
            accuracy = random.uniform(0.70, 0.75)
            precision = random.uniform(0.68, 0.73)
            recall = random.uniform(0.65, 0.72)
        elif name in ['Ridge Regression', 'Lasso Regression']:
            accuracy = random.uniform(0.65, 0.70)
            precision = random.uniform(0.63, 0.68)
            recall = random.uniform(0.60, 0.67)
        else:
            accuracy = random.uniform(0.60, 0.65)
            precision = random.uniform(0.58, 0.63)
            recall = random.uniform(0.55, 0.62)
        
        f1_score = 2 * (precision * recall) / (precision + recall)
        
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
            'feature_names': list(X.columns),
            # 이진 분류 지표
            'classification_threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
        
        results.append(result)
    
    # R2 점수로 정렬
    results.sort(key=lambda x: x['r2_score'], reverse=True)
    
    return results

def save_model(model_info, model_dir='media/models'):
    """
    모델을 파일로 저장합니다. (시연용에서는 간단한 정보만 저장)
    """
    os.makedirs(model_dir, exist_ok=True)
    
    # 고유한 파일명 생성
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{model_info['name'].replace(' ', '_')}_{model_info['target']}_{timestamp}.pkl"
    filepath = os.path.join(model_dir, filename)
    
    # 시연용: 모델 정보만 저장
    save_data = {
        'model_type': model_info['name'],
        'target': model_info['target'],
        'feature_names': model_info['feature_names'],
        'r2_score': model_info['r2_score'],
        'classification_threshold': model_info['classification_threshold']
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
    시연용 반감기 예측 - 랜덤 값 생성
    """
    # 기본 반감기 범위 설정
    if model_data['target'] == 'field':
        base_halflife = random.uniform(10, 60)  # 포장 반감기: 10-60일
        threshold = FIELD_HALFLIFE_THRESHOLD
    else:
        base_halflife = random.uniform(5, 40)   # 실내 반감기: 5-40일
        threshold = LAB_HALFLIFE_THRESHOLD
    
    # 분자 특성에 따라 약간의 조정 (시연 효과)
    if compound.molecular_weight and compound.molecular_weight > 400:
        base_halflife *= 1.2
    if compound.logp and compound.logp > 3:
        base_halflife *= 1.1
    
    # 계통에 따른 조정
    if compound.system == 'Organophosphate':
        base_halflife *= 0.8
    elif compound.system == 'Triazole':
        base_halflife *= 1.3
    
    # 최종 예측값
    predicted_value = max(1, min(100, base_halflife + random.uniform(-5, 5)))
    
    # 신뢰구간 (±15%)
    confidence_interval = predicted_value * 0.15
    
    # 이진 분류 결과
    is_persistent = predicted_value >= threshold
    confidence_score = random.uniform(70, 95) if is_persistent else random.uniform(60, 85)  # 백분율로 변경
    
    return {
        'predicted_value': predicted_value,
        'confidence_lower': max(0, predicted_value - confidence_interval),
        'confidence_upper': predicted_value + confidence_interval,
        'input_features': {
            'molecular_weight': compound.molecular_weight or 0,
            'logp': compound.logp or 0,
            'system': compound.system or 'Unknown'
        },
        # 이진 분류 결과
        'is_persistent': is_persistent,
        'persistence_probability': confidence_score,  # 이미 백분율
        'classification_threshold': threshold,
        'classification': '지속성 높음' if is_persistent else '지속성 낮음'
    }