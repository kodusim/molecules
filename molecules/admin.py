from django.contrib import admin
from .models import Compound, DataUpload, MLModel, Prediction

@admin.register(Compound)
class CompoundAdmin(admin.ModelAdmin):
    list_display = ['name', 'smiles', 'system', 'molecular_weight', 'field_halflife', 'lab_halflife', 'created_at']
    list_filter = ['system', 'created_at']
    search_fields = ['name', 'smiles']
    readonly_fields = ['id', 'created_at', 'updated_at', 'molecule_image']
    
    fieldsets = (
        ('기본 정보', {
            'fields': ('name', 'smiles', 'system')
        }),
        ('분자 특성', {
            'fields': ('molecular_weight', 'logp', 'tpsa', 'num_h_donors', 
                      'num_h_acceptors', 'num_rotatable_bonds', 'num_aromatic_rings', 
                      'num_heavy_atoms'),
            'classes': ('collapse',)
        }),
        ('제품 정보', {
            'fields': ('active_ingredient_content', 'formulation')
        }),
        ('반감기 데이터', {
            'fields': ('field_halflife', 'lab_halflife')
        }),
        ('이미지', {
            'fields': ('molecule_image',)
        }),
        ('메타 정보', {
            'fields': ('id', 'created_by', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )

@admin.register(DataUpload)
class DataUploadAdmin(admin.ModelAdmin):
    list_display = ['file_name', 'uploaded_by', 'uploaded_at', 'status', 'total_rows', 'processed_rows']
    list_filter = ['status', 'uploaded_at']
    readonly_fields = ['id', 'uploaded_at', 'total_rows', 'processed_rows']

@admin.register(MLModel)
class MLModelAdmin(admin.ModelAdmin):
    list_display = ['name', 'model_type', 'target', 'r2_score', 'rmse', 'training_date', 'is_active']
    list_filter = ['model_type', 'target', 'is_active', 'training_date']
    readonly_fields = ['id', 'training_date', 'feature_names', 'feature_importances']

@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ['compound', 'model', 'predicted_value', 'predicted_by', 'predicted_at']
    list_filter = ['model', 'predicted_at']
    readonly_fields = ['id', 'predicted_at', 'input_features']