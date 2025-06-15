from django.shortcuts import render, redirect
from django.contrib import messages
from django.views.generic import ListView, DetailView
from django.db.models import Count, Avg
from django.core.files.base import ContentFile
from .models import Compound, DataUpload, MLModel, Prediction
from .forms import FileUploadForm
from .utils import process_uploaded_file, calculate_molecular_properties, generate_molecule_image
from .ml_utils import prepare_training_data, train_models, save_model, load_model, predict_halflife
import os

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
    
    context = {
        'models': models,
        'system_stats': system_stats,
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
    # 학습 가능한 데이터 수 확인
    field_count = Compound.objects.exclude(field_halflife__isnull=True).count()
    lab_count = Compound.objects.exclude(lab_halflife__isnull=True).count()
    
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
        
        # 데이터 준비
        if target == 'field':
            compounds = Compound.objects.exclude(field_halflife__isnull=True)
        else:
            compounds = Compound.objects.exclude(lab_halflife__isnull=True)
        
        if compounds.count() < 10:
            messages.error(request, '학습에 필요한 데이터가 부족합니다. (최소 10개 필요)')
            return redirect('molecules:model_training')
        
        try:
            # 학습 데이터 준비
            X, y, feature_names = prepare_training_data(compounds, target)
            
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