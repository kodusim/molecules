from django.urls import path
from . import views

app_name = 'molecules'

urlpatterns = [
    path('', views.index, name='index'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('compounds/', views.CompoundListView.as_view(), name='compound_list'),
    path('compounds/<uuid:pk>/', views.CompoundDetailView.as_view(), name='compound_detail'),
]