from django.urls import path
from . import views

app_name = 'molecules'

urlpatterns = [
    # 홈페이지
    path('', views.index, name='index'),
    
    # 대시보드
    path('dashboard/', views.dashboard, name='dashboard'),
    
    # 화합물 관련
    path('compounds/', views.CompoundListView.as_view(), name='compound_list'),
    path('compounds/<uuid:pk>/', views.CompoundDetailView.as_view(), name='compound_detail'),
    
    # 파일 업로드
    path('upload/', views.file_upload, name='file_upload'),
    path('uploads/', views.upload_list, name='upload_list'),
    
    # 모델 학습
    path('models/training/', views.model_training, name='model_training'),
    path('models/train/', views.train_model, name='train_model'),
    path('models/', views.model_list, name='model_list'),
    
    # 예측 관련
    path('predict/', views.predict, name='predict'),
]