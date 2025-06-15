import pandas as pd
import numpy as np
import io
from PIL import Image
import base64

def calculate_molecular_properties(smiles):
    """
    SMILES 문자열로부터 분자 특성을 계산합니다.
    (RDKit 없이 더미 데이터 반환)
    """
    # RDKit가 없으므로 더미 데이터 반환
    # 실제 배포 시에는 API나 다른 방법을 사용할 수 있습니다
    np.random.seed(hash(smiles) % 2**32)
    
    properties = {
        'molecular_weight': round(200 + np.random.random() * 400, 2),
        'logp': round(np.random.random() * 5, 2),
        'tpsa': round(40 + np.random.random() * 100, 2),
        'num_h_donors': int(np.random.randint(0, 5)),
        'num_h_acceptors': int(np.random.randint(0, 10)),
        'num_rotatable_bonds': int(np.random.randint(0, 15)),
        'num_aromatic_rings': int(np.random.randint(0, 4)),
        'num_heavy_atoms': int(np.random.randint(10, 50)),
    }
    
    return properties

def generate_molecule_image(smiles, size=(300, 300)):
    """
    SMILES로부터 분자 구조 이미지를 생성합니다.
    (RDKit 없이 플레이스홀더 이미지 생성)
    """
    try:
        # 플레이스홀더 이미지 생성
        img = Image.new('RGB', size, color='white')
        
        # PIL Image를 bytes로 변환
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        return img_buffer
    except Exception as e:
        print(f"Error generating placeholder image: {e}")
        return None

def process_uploaded_file(file_path, file_type='excel'):
    """
    업로드된 파일을 처리합니다.
    """
    try:
        # 파일 읽기
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, encoding='utf-8-sig')
        else:
            df = pd.read_excel(file_path)
        
        # 열 이름 정리 (공백 제거)
        df.columns = df.columns.str.strip()
        
        # SMILES 열 찾기
        smiles_col = None
        for col in df.columns:
            if 'SMILES' in col.upper() or 'SMILE' in col.upper():
                smiles_col = col
                break
        
        if not smiles_col:
            raise ValueError("SMILES 열을 찾을 수 없습니다.")
        
        # 열 이름 표준화
        column_mapping = {smiles_col: 'SMILES'}
        
        # 다른 열들 매핑
        for col in df.columns:
            if '성분명' in col or '화합물명' in col:
                column_mapping[col] = 'name'
            elif '포장' in col and '반감기' in col:
                column_mapping[col] = 'field_halflife'
            elif '실내' in col and '반감기' in col:
                column_mapping[col] = 'lab_halflife'
            elif '계통' in col or 'system' in col.lower():
                column_mapping[col] = 'system'
            elif '활성' in col and '함량' in col:
                column_mapping[col] = 'active_ingredient_content'
            elif '제형' in col:
                column_mapping[col] = 'formulation'
        
        df_renamed = df.rename(columns=column_mapping)
        
        # SMILES 유효성 검사 (간단한 검사만)
        valid_rows = []
        for idx, row in df_renamed.iterrows():
            smiles = str(row['SMILES']).strip()
            if pd.notna(smiles) and smiles and len(smiles) > 0:
                # RDKit 없이는 실제 유효성 검사가 어려우므로
                # 간단한 문자열 검사만 수행
                if 'C' in smiles or 'c' in smiles:  # 최소한 탄소가 있어야 함
                    valid_rows.append(idx)
        
        df_valid = df_renamed.loc[valid_rows]
        
        return df_valid, len(df), len(df_valid)
        
    except Exception as e:
        raise Exception(f"파일 처리 중 오류 발생: {str(e)}")

def prepare_features_for_ml(compounds):
    """
    화합물 데이터를 머신러닝용 특성 행렬로 변환합니다.
    """
    features = []
    
    for compound in compounds:
        # 분자 특성
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
        if compound.system:
            system_features = {
                f'system_{sys}': 1 if compound.system == sys else 0
                for sys in ['Organophosphate', 'Triazole', 'Carbamate', 'Amide', 
                           'Sulfonylurea', 'Anilide', 'Strobilurin', 'Pyrazole']
            }
            feature_dict.update(system_features)
        
        features.append(feature_dict)
    
    return pd.DataFrame(features)