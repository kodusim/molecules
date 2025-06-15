from django.urls import path
from . import views

app_name = 'molecules'

urlpatterns = [
    path('', views.index, name='index'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('compounds/', views.CompoundListView.as_view(), name='compound_list'),
    path('compounds/<uuid:pk>/', views.CompoundDetailView.as_view(), name='compound_detail'),
    path('upload/', views.file_upload, name='file_upload'),
    path('uploads/', views.upload_list, name='upload_list'),
    path('model-training/', views.model_training, name='model_training'),
    path('train-model/', views.train_model, name='train_model'),
    path('models/', views.model_list, name='model_list'),
]