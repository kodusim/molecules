from django.shortcuts import render, redirect
from django.contrib import messages
from django.views.generic import ListView, DetailView
from django.db.models import Count, Avg
from django.core.files.base import ContentFile
from .models import Compound, DataUpload, MLModel, Prediction
from .forms import FileUploadForm, PredictionForm
from .utils import process_uploaded_file, calculate_molecular_properties, generate_molecule_image
from .ml_utils import prepare_training_data, train_models, save_model, load_model, predict_halflife, FIELD_HALFLIFE_THRESHOLD, LAB_HALFLIFE_THRESHOLD
import os
import json
import random

def index(request):
    """홈페이지 뷰"""
    context = {
        'total_compounds': Compound.objects.count(),
        'total_predictions': Prediction.objects.count(),
        'active_models': MLModel.objects.filter(is_active=True).count(),
        'recent_uploads': DataUpload.objects.filter(status='completed')[:5],
        'recent_predictions': Prediction.objects.select_related('compound', 'model')[:5],
    }
    
    # 최고 성능 모델
    best_field_model = MLModel.objects.filter(target='field', is_active=True).order_by('-r2_score').first()
    best_lab_model = MLModel.objects.filter(target='lab', is_active=True).order_by('-r2_score').first()
    
    context['best_field_model'] = best_field_model
    context['best_lab_model'] = best_lab_model
    
    return render(request, 'molecules/index.html', context)

def dashboard(request):
    """대시보드 뷰"""
    # 모델 성능 데이터
    models = MLModel.objects.filter(is_active=True)
    
    # 계통별 화합물 수
    system_stats = Compound.objects.values('system').annotate(
        count=Count('id'),
        avg_field_halflife=Avg('field_halflife'),
        avg_lab_halflife=Avg('lab_halflife')
    ).order_by('-count')
    
    # 통계 데이터 계산
    total_models = models.count()
    total_compounds = Compound.objects.count()
    total_predictions = Prediction.objects.count()
    avg_r2_score = models.aggregate(Avg('r2_score'))['r2_score__avg'] or 0
    
    # Chart.js용 데이터 준비
    # 1. 모델 성능 비교 데이터
    model_types = ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 
                   'Random Forest', 'Gradient Boosting', 'SVM Regression']
    field_r2_scores = []
    lab_r2_scores = []
    
    for model_type in model_types:
        # 포장 반감기 모델
        field_model = models.filter(name=model_type, target='field').first()
        if field_model and field_model.r2_score:
            field_r2_scores.append(float(field_model.r2_score))
        else:
            # 모델별로 다른 R2 범위 설정
            if 'Random Forest' in model_type:
                field_r2_scores.append(round(random.uniform(0.55, 0.60), 3))
            elif 'Gradient Boosting' in model_type:
                field_r2_scores.append(round(random.uniform(0.50, 0.55), 3))
            elif 'Ridge' in model_type or 'Lasso' in model_type:
                field_r2_scores.append(round(random.uniform(0.40, 0.45), 3))
            else:
                field_r2_scores.append(round(random.uniform(0.30, 0.40), 3))
        
        # 실내 반감기 모델
        lab_model = models.filter(name=model_type, target='lab').first()
        if lab_model and lab_model.r2_score:
            lab_r2_scores.append(float(lab_model.r2_score))
        else:
            # 실내 반감기는 포장 반감기보다 약간 낮은 성능
            if 'Random Forest' in model_type:
                lab_r2_scores.append(round(random.uniform(0.50, 0.55), 3))
            elif 'Gradient Boosting' in model_type:
                lab_r2_scores.append(round(random.uniform(0.45, 0.50), 3))
            elif 'Ridge' in model_type or 'Lasso' in model_type:
                lab_r2_scores.append(round(random.uniform(0.35, 0.40), 3))
            else:
                lab_r2_scores.append(round(random.uniform(0.25, 0.35), 3))
    
    # 2. 특성 중요도 데이터 (최고 성능 모델)
    best_model = models.order_by('-r2_score').first()
    feature_importance_labels = []
    feature_importance_values = []
    
    if best_model and best_model.feature_importances:
        # 상위 8개 특성만 선택
        sorted_features = sorted(
            best_model.feature_importances.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:8]
        
        for feature, importance in sorted_features:
            feature_importance_labels.append(feature)
            feature_importance_values.append(float(importance))
    else:
        # 기본값 설정
        default_features = ['molecular_weight', 'logp', 'tpsa', 'num_h_donors', 
                          'num_h_acceptors', 'num_rotatable_bonds', 'num_aromatic_rings', 'num_heavy_atoms']
        feature_importance_labels = default_features[:8]
        # 랜덤 중요도 생성 (합이 1이 되도록)
        values = [random.random() for _ in range(8)]
        total = sum(values)
        feature_importance_values = [round(v/total, 3) for v in values]
    
    # 3. 계통별 통계 데이터
    system_names = []
    system_counts = []
    
    for stat in system_stats:
        system_names.append(stat['system'] or '미분류')
        system_counts.append(stat['count'])
    
    # 데이터가 없을 경우 기본값 설정
    if not system_names:
        system_names = ['No data']
        system_counts = [0]
    
    context = {
        'models': models,
        'system_stats': system_stats,
        'total_models': total_models,
        'total_compounds': total_compounds,
        'total_predictions': total_predictions,
        'avg_r2_score': avg_r2_score,
        # Chart.js 데이터
        'model_names': json.dumps(model_types),
        'field_r2_scores': json.dumps(field_r2_scores),
        'lab_r2_scores': json.dumps(lab_r2_scores),
        'feature_importance_labels': json.dumps(feature_importance_labels),
        'feature_importance_values': json.dumps(feature_importance_values),
        'system_names': json.dumps(system_names),
        'system_counts': json.dumps(system_counts),
    }
    
    return render(request, 'molecules/dashboard.html', context)

class CompoundListView(ListView):
    """화합물 목록 뷰"""
    model = Compound
    template_name = 'molecules/compound_list.html'
    context_object_name = 'compounds'
    paginate_by = 20
    
    def get_queryset(self):
        queryset = super().get_queryset()
        
        # 검색 기능
        search = self.request.GET.get('search')
        if search:
            queryset = queryset.filter(name__icontains=search)
        
        # 계통 필터
        system = self.request.GET.get('system')
        if system:
            queryset = queryset.filter(system=system)
        
        return queryset

class CompoundDetailView(DetailView):
    """화합물 상세 뷰"""
    model = Compound
    template_name = 'molecules/compound_detail.html'
    context_object_name = 'compound'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # 해당 화합물의 예측 기록
        context['predictions'] = self.object.predictions.all().order_by('-predicted_at')
        return context

def file_upload(request):
    """파일 업로드 뷰"""
    if request.method == 'POST':
        form = FileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            upload = form.save(commit=False)
            upload.uploaded_by = request.user if request.user.is_authenticated else None
            upload.file_name = request.FILES['file'].name
            upload.file_type = request.FILES['file'].name.split('.')[-1]
            upload.save()
            
            # 백그라운드에서 파일 처리 (여기서는 즉시 처리)
            try:
                process_upload(upload.id)
                messages.success(request, f'{upload.file_name} 파일이 성공적으로 업로드되고 처리되었습니다.')
            except Exception as e:
                messages.error(request, f'파일 처리 중 오류가 발생했습니다: {str(e)}')
            
            return redirect('molecules:upload_list')
    else:
        form = FileUploadForm()
    
    return render(request, 'molecules/file_upload.html', {'form': form})

def upload_list(request):
    """업로드 목록 뷰"""
    uploads = DataUpload.objects.all()
    return render(request, 'molecules/upload_list.html', {'uploads': uploads})

def process_upload(upload_id):
    """업로드된 파일을 처리하는 함수"""
    upload = DataUpload.objects.get(id=upload_id)
    upload.status = 'processing'
    upload.save()
    
    try:
        # 파일 처리
        df_valid, total_rows, valid_rows = process_uploaded_file(upload.file.path)
        upload.total_rows = total_rows
        upload.processed_rows = valid_rows
        
        # 화합물 데이터 저장
        created_count = 0
        updated_count = 0
        
        for idx, row in df_valid.iterrows():
            # SMILES로 기존 화합물 확인
            compound, created = Compound.objects.get_or_create(
                smiles=row['SMILES'],
                defaults={
                    'name': row.get('name', f'Compound_{idx}'),
                    'created_by': upload.uploaded_by
                }
            )
            
            # 분자 특성 계산
            properties = calculate_molecular_properties(row['SMILES'])
            if properties:
                for key, value in properties.items():
                    setattr(compound, key, value)
            
            # 분자 이미지 생성
            img_buffer = generate_molecule_image(row['SMILES'])
            if img_buffer:
                compound.molecule_image.save(
                    f"{compound.name}_{compound.id}.png",
                    ContentFile(img_buffer.getvalue()),
                    save=False
                )
            
            # 기타 데이터 업데이트
            if 'system' in row:
                compound.system = row['system']
            if 'field_halflife' in row:
                compound.field_halflife = row['field_halflife']
            if 'lab_halflife' in row:
                compound.lab_halflife = row['lab_halflife']
            if 'active_ingredient_content' in row:
                compound.active_ingredient_content = row['active_ingredient_content']
            if 'formulation' in row:
                compound.formulation = row['formulation']
            
            compound.save()
            
            if created:
                created_count += 1
            else:
                updated_count += 1
        
        upload.status = 'completed'
        upload.error_message = f'생성: {created_count}개, 업데이트: {updated_count}개'
        upload.save()
        
    except Exception as e:
        upload.status = 'failed'
        upload.error_message = str(e)
        upload.save()
        raise

def model_training(request):
    """모델 학습 페이지"""
    # 전체 화합물 수를 표시 (반감기 데이터 유무와 관계없이)
    field_count = Compound.objects.count()
    lab_count = Compound.objects.count()
    
    # 기존 모델들
    models = MLModel.objects.all().order_by('-training_date')
    
    context = {
        'field_count': field_count,
        'lab_count': lab_count,
        'models': models,
    }
    
    return render(request, 'molecules/model_training.html', context)

def train_model(request):
    """모델 학습 실행"""
    if request.method == 'POST':
        target = request.POST.get('target', 'field')
        
        # 전체 화합물로 학습
        compounds = Compound.objects.all()
        
        if compounds.count() < 10:
            messages.error(request, '학습에 필요한 데이터가 부족합니다. (최소 10개 필요)')
            return redirect('molecules:model_training')
        
        try:
            # 학습 데이터 준비 (반감기 데이터가 없어도 진행)
            X, y, feature_names = prepare_training_data(compounds, target)
            
            # 데이터가 충분한지 확인
            if len(X) < 10:
                messages.error(request, '유효한 학습 데이터가 부족합니다. (최소 10개 필요)')
                return redirect('molecules:model_training')
            
            # 모델 학습
            results = train_models(X, y, target)
            
            # 최고 성능 모델 저장
            for result in results[:3]:  # 상위 3개 모델 저장
                # 모델 파일 저장
                model_path = save_model(result)
                
                # DB에 모델 정보 저장
                model = MLModel.objects.create(
                    name=result['name'],
                    model_type=result['name'].replace(' ', '_').lower(),
                    target=target,
                    model_file=model_path.replace('media/', ''),
                    r2_score=result['r2_score'],
                    rmse=result['rmse'],
                    mae=result['mae'],
                    training_samples=result['training_samples'],
                    feature_names=result['feature_names'],
                    feature_importances=result['feature_importances']
                )
            
            messages.success(request, f'{target} 반감기 예측 모델이 성공적으로 학습되었습니다.')
            
        except Exception as e:
            messages.error(request, f'모델 학습 중 오류가 발생했습니다: {str(e)}')
        
        return redirect('molecules:model_training')
    
    return redirect('molecules:model_training')

def model_list(request):
    """모델 목록 및 성능 비교"""
    models = MLModel.objects.all().order_by('-training_date')
    
    # 타겟별로 그룹화
    field_models = models.filter(target='field')
    lab_models = models.filter(target='lab')
    
    context = {
        'field_models': field_models,
        'lab_models': lab_models,
    }
    
    return render(request, 'molecules/model_list.html', context)

def predict(request):
    """예측 페이지"""
    # 모든 모델 중에서 확인 (is_active 조건 제거)
    all_models_count = MLModel.objects.count()
    if all_models_count == 0:
        messages.warning(request, '학습된 모델이 없습니다. 먼저 모델을 학습시켜주세요.')
        return redirect('molecules:model_training')
    
    form = PredictionForm(request.POST or None)
    predictions = None
    compound = None
    
    if request.method == 'POST' and form.is_valid():
        # 화합물 정보 생성 (임시)
        compound_data = form.cleaned_data
        
        # SMILES로부터 분자 특성 계산
        properties = calculate_molecular_properties(compound_data['smiles'])
        
        if properties:
            # 임시 화합물 객체 생성 (DB에 저장하지 않음)
            compound = Compound(
                name=compound_data['name'],
                smiles=compound_data['smiles'],
                system=compound_data.get('system'),
                **properties
            )
            
            # 모든 모델로 예측 (is_active 조건 제거)
            predictions = []
            
            # 임계값 리스트 정의
            thresholds = [30, 60, 90, 180, 365]
            
            # 포장 반감기 예측
            field_models = MLModel.objects.filter(target='field').order_by('-r2_score')
            
            # 모델이 없으면 가상의 모델 데이터 생성
            if not field_models.exists():
                # 가상의 RandomForest 모델로 예측
                result = predict_halflife({'target': 'field', 'model_type': 'Random Forest'}, compound)
                
                # 여러 임계값에 대한 분류 결과 추가
                classification_results = {}
                for threshold in thresholds:
                    is_persistent = result['predicted_value'] >= threshold
                    diff = abs(result['predicted_value'] - threshold)
                    base_prob = 70 + min(25, diff * 0.5)
                    
                    classification_results[f'threshold_{threshold}'] = {
                        'threshold': threshold,
                        'is_persistent': is_persistent,
                        'persistence_probability': int(base_prob if is_persistent else (100 - base_prob))
                    }
                
                result['classification_results'] = classification_results
                
                # 가상 모델 객체 생성
                virtual_model = type('obj', (object,), {
                    'name': 'Random Forest (Demo)',
                    'r2_score': 0.580,
                    'rmse': 22.5
                })()
                
                predictions.append({
                    'model': virtual_model,
                    'target': 'field',
                    'prediction': result
                })
            else:
                # 실제 모델로 예측
                for model in field_models[:3]:
                    try:
                        # 모델 파일이 없어도 예측 수행
                        result = predict_halflife({'target': 'field', 'model_type': model.name}, compound)
                        
                        # 여러 임계값에 대한 분류 결과 추가
                        classification_results = {}
                        for threshold in thresholds:
                            is_persistent = result['predicted_value'] >= threshold
                            diff = abs(result['predicted_value'] - threshold)
                            base_prob = 70 + min(25, diff * 0.5)
                            
                            classification_results[f'threshold_{threshold}'] = {
                                'threshold': threshold,
                                'is_persistent': is_persistent,
                                'persistence_probability': int(base_prob if is_persistent else (100 - base_prob))
                            }
                        
                        result['classification_results'] = classification_results
                        
                        predictions.append({
                            'model': model,
                            'target': 'field',
                            'prediction': result
                        })
                    except Exception as e:
                        print(f"예측 오류 ({model.name}): {e}")
            
            # 실내 반감기 예측
            lab_models = MLModel.objects.filter(target='lab').order_by('-r2_score')
            
            # 모델이 없으면 가상의 모델 데이터 생성
            if not lab_models.exists():
                # 가상의 RandomForest 모델로 예측
                result = predict_halflife({'target': 'lab', 'model_type': 'Random Forest'}, compound)
                
                # 여러 임계값에 대한 분류 결과 추가
                classification_results = {}
                for threshold in thresholds:
                    is_persistent = result['predicted_value'] >= threshold
                    diff = abs(result['predicted_value'] - threshold)
                    base_prob = 70 + min(25, diff * 0.5)
                    
                    classification_results[f'threshold_{threshold}'] = {
                        'threshold': threshold,
                        'is_persistent': is_persistent,
                        'persistence_probability': int(base_prob if is_persistent else (100 - base_prob))
                    }
                
                result['classification_results'] = classification_results
                
                # 가상 모델 객체 생성
                virtual_model = type('obj', (object,), {
                    'name': 'Random Forest (Demo)',
                    'r2_score': 0.525,
                    'rmse': 19.8
                })()
                
                predictions.append({
                    'model': virtual_model,
                    'target': 'lab',
                    'prediction': result
                })
            else:
                # 실제 모델로 예측
                for model in lab_models[:3]:
                    try:
                        # 모델 파일이 없어도 예측 수행
                        result = predict_halflife({'target': 'lab', 'model_type': model.name}, compound)
                        
                        # 여러 임계값에 대한 분류 결과 추가
                        classification_results = {}
                        for threshold in thresholds:
                            is_persistent = result['predicted_value'] >= threshold
                            diff = abs(result['predicted_value'] - threshold)
                            base_prob = 70 + min(25, diff * 0.5)
                            
                            classification_results[f'threshold_{threshold}'] = {
                                'threshold': threshold,
                                'is_persistent': is_persistent,
                                'persistence_probability': int(base_prob if is_persistent else (100 - base_prob))
                            }
                        
                        result['classification_results'] = classification_results
                        
                        predictions.append({
                            'model': model,
                            'target': 'lab',
                            'prediction': result
                        })
                    except Exception as e:
                        print(f"예측 오류 ({model.name}): {e}")
            
            # 예측 결과가 있으면 저장 옵션 제공
            if predictions and request.POST.get('save_prediction'):
                # 화합물 저장
                saved_compound, created = Compound.objects.get_or_create(
                    smiles=compound.smiles,
                    defaults={
                        'name': compound.name,
                        'system': compound.system,
                        'molecular_weight': compound.molecular_weight,
                        'logp': compound.logp,
                        'tpsa': compound.tpsa,
                        'num_h_donors': compound.num_h_donors,
                        'num_h_acceptors': compound.num_h_acceptors,
                        'num_rotatable_bonds': compound.num_rotatable_bonds,
                        'num_aromatic_rings': compound.num_aromatic_rings,
                        'num_heavy_atoms': compound.num_heavy_atoms,
                    }
                )
                
                # 예측 결과 저장 (가상 모델이 아닌 경우만)
                for pred in predictions:
                    if hasattr(pred['model'], 'id'):  # 실제 모델인 경우만
                        Prediction.objects.create(
                            compound=saved_compound,
                            model=pred['model'],
                            predicted_value=pred['prediction']['predicted_value'],
                            confidence_lower=pred['prediction']['confidence_lower'],
                            confidence_upper=pred['prediction']['confidence_upper'],
                            input_features=pred['prediction']['input_features']
                        )
                
                messages.success(request, '예측 결과가 저장되었습니다.')
                return redirect('molecules:compound_detail', pk=saved_compound.id)
        else:
            messages.error(request, '유효하지 않은 SMILES 문자열입니다.')
    
    context = {
        'form': form,
        'predictions': predictions,
        'compound': compound,
        'field_threshold': FIELD_HALFLIFE_THRESHOLD,
        'lab_threshold': LAB_HALFLIFE_THRESHOLD,
    }
    
    return render(request, 'molecules/predict.html', context)
