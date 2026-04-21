import re
from typing import Any

from .load import load

def from_desc(desc: str) -> Any:
    class_name = desc.split("(")[0]
    kwargs = dict(kwarg.split('=') for kwarg in desc.split('(')[1].replace(')', '').split(', '))
    kwargs = {k: eval(v) for k, v in kwargs.items()}
    print(class_name)
    print(kwargs)

    # Get class type.
    if class_name.endswith('Dataset'):
        class_type = 'dataset'
    elif class_name.endswith('Patient'):
        class_type = 'patient'
    elif class_name.endswith('Study'):
        class_type = 'study'
    elif class_name.endswith('Series'):
        class_type = 'series'
    else:
        raise ValueError(f"Unrecognised class {class_name}")
    print(class_type)

    # Get the dataset class.
    class_parts = re.split(r'(?=[A-Z])', class_name)
    dataset_class_name = class_parts[1]
    if not dataset_class_name.lower() in ['dicom', 'nifti']:
        raise ValueError(f"Unrecognised dataset class {dataset_class_name}")

    # Load dataset.
    dataset_id_kwarg = 'id' if class_type == 'dataset' else 'dataset_id'
    dataset = load(kwargs[dataset_id_kwarg], dataset_class_name)
    if class_type == 'dataset':
        assert str(dataset) == desc
        return dataset

    # Load patient.
    patient_id_kwarg = 'id' if class_type == 'patient' else 'patient_id'
    patient = dataset.patient(kwargs[patient_id_kwarg])
    if class_type == 'patient':
        assert str(patient) == desc 
        return patient

    # Load study.
    study_id_kwarg = 'id' if class_type == 'study' else 'study_id'
    study = patient.study(kwargs[study_id_kwarg])
    if class_type == 'study':
        assert str(study) == desc
        return study

    # Load series.
    modality = ''.join(class_parts[2:-1]).lower()
    print(modality)
    series = study.series(kwargs['id'], modality)
    assert str(series) == desc
    return series
