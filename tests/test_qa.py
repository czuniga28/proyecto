import pytest

from src.data.qa import run_qa_checks


def test_clean_manifest_has_empty_report(manifest):
    report = run_qa_checks(manifest)

    assert report.empty
    assert list(report.columns) == ['examId', 'seriesNo', 'check', 'detail']


@pytest.mark.parametrize("column, value, check", [
    ('aclDiagnosis', 7, 'label_validity'),
    ('kneeLR', 3, 'label_validity'),
    ('roiDepth', 0, 'roi_size'),
    ('roiX', 400, 'roi_bounds'),
    ('roiY', -1, 'roi_bounds'),
    ('roiZ', 29, 'roi_bounds'),       # 29 + 3 > 30 slices
    ('dtype', 'float32', 'dtype'),
    ('intensity_max', 0, 'intensity'),
    ('intensity_max', 5000, 'intensity'),
    ('exists', False, 'file_exists'),
])
def test_each_check_can_fail(manifest, column, value, check):
    bad = manifest.copy()
    bad.loc[0, column] = value

    report = run_qa_checks(bad)

    assert len(report) == 1
    assert report.loc[0, 'check'] == check
    assert report.loc[0, 'examId'] == manifest.loc[0, 'examId']


def test_bounds_detail_names_the_axis(manifest):
    bad = manifest.copy()
    bad.loc[0, 'roiX'] = 400

    detail = run_qa_checks(bad).loc[0, 'detail']

    assert detail.startswith('x:')
