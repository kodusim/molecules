from django.db import models
from django.contrib.auth.models import User
from django.core.validators import FileExtensionValidator
import uuid
import json

class Compound(models.Model):
    """화합물 정보를 저장하는 모델"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200, verbose_name='화합물명')
    smiles = models.TextField(verbose_name='SMILES', unique=True)
    
    # 계통 정보
    SYSTEM_CHOICES = [
        ('Organophosphate', 'Organophosphate'),
        ('Triazole', 'Triazole'),
        ('Carbamate', 'Carbamate'),
        ('Amide', 'Amide'),
        ('Sulfonylurea', 'Sulfonylurea'),
        ('Anilide', 'Anilide'),
        ('Strobilurin', 'Strobilurin'),
        ('Pyrazole', 'Pyrazole'),
        ('Other', '기타'),
    ]
    system = models.CharField(max_length=50, choices=SYSTEM_CHOICES, verbose_name='계통', null=True, blank=True)
    
    # 분자 특성 (자동 계산)
    molecular_weight = models.FloatField(null=True, blank=True, verbose_name='분자량')
    logp = models.FloatField(null=True, blank=True, verbose_name='LogP')
    tpsa = models.FloatField(null=True, blank=True, verbose_name='TPSA')
    num_h_donors = models.IntegerField(null=True, blank=True, verbose_name='수소 공여체 수')
    num_h_acceptors = models.IntegerField(null=True, blank=True, verbose_name='수소 수용체 수')
    num_rotatable_bonds = models.IntegerField(null=True, blank=True, verbose_name='회전 가능 결합 수')
    num_aromatic_rings = models.IntegerField(null=True, blank=True, verbose_name='방향족 고리 수')
    num_heavy_atoms = models.IntegerField(null=True, blank=True, verbose_name='무거운 원자 수')
    
    # 제품 정보
    active_ingredient_content = models.FloatField(null=True, blank=True, verbose_name='활성성분함량(%)')
    formulation = models.CharField(max_length=100, null=True, blank=True, verbose_name='제형')
    
    # 반감기 데이터
    field_halflife = models.FloatField(null=True, blank=True, verbose_name='포장반감기(일)')
    lab_halflife = models.FloatField(null=True, blank=True, verbose_name='실내반감기(일)')
    
    # 분자 이미지
    molecule_image = models.ImageField(upload_to='molecules/', null=True, blank=True, verbose_name='분자 구조 이미지')
    
    # 메타 정보
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, verbose_name='작성자')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='생성일')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='수정일')
    
    class Meta:
        verbose_name = '화합물'
        verbose_name_plural = '화합물'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} ({self.smiles[:20]}...)" if len(self.smiles) > 20 else f"{self.name} ({self.smiles})"

class DataUpload(models.Model):
    """데이터 업로드 기록"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True, verbose_name='업로드한 사용자')
    uploaded_at = models.DateTimeField(auto_now_add=True, verbose_name='업로드 시간')
    
    # 파일 정보
    file = models.FileField(
        upload_to='uploads/%Y/%m/%d/',
        validators=[FileExtensionValidator(['xlsx', 'xls', 'csv'])],
        verbose_name='업로드 파일'
    )
    file_name = models.CharField(max_length=255, verbose_name='파일명')
    file_type = models.CharField(max_length=50, verbose_name='파일 타입')
    
    # 처리 상태
    STATUS_CHOICES = [
        ('pending', '대기중'),
        ('processing', '처리중'),
        ('completed', '완료'),
        ('failed', '실패'),
    ]
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending', verbose_name='상태')
    
    # 처리 결과
    total_rows = models.IntegerField(default=0, verbose_name='전체 행 수')
    processed_rows = models.IntegerField(default=0, verbose_name='처리된 행 수')
    error_message = models.TextField(blank=True, verbose_name='오류 메시지')
    
    class Meta:
        verbose_name = '데이터 업로드'
        verbose_name_plural = '데이터 업로드'
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return f"{self.file_name} - {self.get_status_display()}"

class MLModel(models.Model):
    """머신러닝 모델 정보"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100, verbose_name='모델명')
    model_type = models.CharField(max_length=50, verbose_name='모델 타입')
    target = models.CharField(max_length=50, verbose_name='예측 대상')  # 'field' or 'lab'
    
    # 모델 파일
    model_file = models.FileField(upload_to='models/', verbose_name='모델 파일')
    
    # 성능 지표
    r2_score = models.FloatField(null=True, blank=True, verbose_name='R² 점수')
    rmse = models.FloatField(null=True, blank=True, verbose_name='RMSE')
    mae = models.FloatField(null=True, blank=True, verbose_name='MAE')
    
    # 학습 정보
    training_date = models.DateTimeField(auto_now_add=True, verbose_name='학습 날짜')
    training_samples = models.IntegerField(default=0, verbose_name='학습 샘플 수')
    feature_names = models.JSONField(default=list, verbose_name='특성 이름들')
    feature_importances = models.JSONField(default=dict, verbose_name='특성 중요도')
    
    # 활성화 여부
    is_active = models.BooleanField(default=True, verbose_name='활성화')
    
    class Meta:
        verbose_name = 'ML 모델'
        verbose_name_plural = 'ML 모델'
        ordering = ['-training_date']
    
    def __str__(self):
        return f"{self.name} - {self.target} ({self.r2_score:.3f})"

class Prediction(models.Model):
    """예측 기록"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    compound = models.ForeignKey(Compound, on_delete=models.CASCADE, related_name='predictions', verbose_name='화합물')
    model = models.ForeignKey(MLModel, on_delete=models.CASCADE, verbose_name='사용된 모델')
    
    # 예측 결과
    predicted_value = models.FloatField(verbose_name='예측값')
    confidence_lower = models.FloatField(null=True, blank=True, verbose_name='신뢰구간 하한')
    confidence_upper = models.FloatField(null=True, blank=True, verbose_name='신뢰구간 상한')
    
    # 예측 정보
    predicted_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, verbose_name='예측한 사용자')
    predicted_at = models.DateTimeField(auto_now_add=True, verbose_name='예측 시간')
    
    # 입력된 특성값들 (디버깅용)
    input_features = models.JSONField(default=dict, verbose_name='입력 특성값')
    
    class Meta:
        verbose_name = '예측'
        verbose_name_plural = '예측'
        ordering = ['-predicted_at']
    
    def __str__(self):
        return f"{self.compound.name} - {self.predicted_value:.2f}일"