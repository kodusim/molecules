from django import forms
from .models import DataUpload, Compound

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

class PredictionForm(forms.Form):
    """화합물 반감기 예측을 위한 폼 (제형과 활성성분함량 제거)"""
    name = forms.CharField(
        max_length=200,
        label='화합물명',
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': '예: Atrazine'
        })
    )
    
    smiles = forms.CharField(
        label='SMILES',
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 3,
            'placeholder': '예: CC(C)NCC(O)COc1cccc2c1C(=O)N(C)C2=O'
        }),
        help_text='화학 구조를 나타내는 SMILES 문자열을 입력하세요.'
    )
    
    system = forms.ChoiceField(
        choices=[('', '선택하세요')] + Compound.SYSTEM_CHOICES,
        required=False,
        label='계통',
        widget=forms.Select(attrs={'class': 'form-select'})
    )