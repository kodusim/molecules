from django.shortcuts import render, redirect
from django.contrib import messages
from django.views.generic import ListView, DetailView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.db.models import Count, Avg
from .models import Compound, DataUpload, MLModel, Prediction

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

class CompoundListView(LoginRequiredMixin, ListView):
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

class CompoundDetailView(LoginRequiredMixin, DetailView):
    """화합물 상세 뷰"""
    model = Compound
    template_name = 'molecules/compound_detail.html'
    context_object_name = 'compound'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # 해당 화합물의 예측 기록
        context['predictions'] = self.object.predictions.all().order_by('-predicted_at')
        return context