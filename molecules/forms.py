from django import forms
from .models import DataUpload

class FileUploadForm(forms.ModelForm):
    class Meta:
        model = DataUpload
        fields = ['file']
        widgets = {
            'file': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.xlsx,.xls,.csv'
            })
        }
        
    def clean_file(self):
        file = self.cleaned_data.get('file')
        if file:
            # 파일 확장자 확인
            ext = file.name.split('.')[-1].lower()
            if ext not in ['xlsx', 'xls', 'csv']:
                raise forms.ValidationError('Excel(.xlsx, .xls) 또는 CSV 파일만 업로드 가능합니다.')
            
            # 파일 크기 확인 (10MB 제한)
            if file.size > 10 * 1024 * 1024:
                raise forms.ValidationError('파일 크기는 10MB를 초과할 수 없습니다.')
        
        return file