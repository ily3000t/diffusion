import pytest
from sumodiff.diffusion.stage5b_report import planned_catalog
def cat(n):return [dict(window_id=f'w{i}') for i in range(n)]
def test_full_contract():
    assert len(planned_catalog(cat(200),None))==200
    with pytest.raises(ValueError):planned_catalog(cat(60),None)
def test_explicit_prefix():
    ids=[f'w{i}' for i in range(60)]
    assert planned_catalog(cat(200),ids)==cat(60)
    assert planned_catalog(cat(60),ids)==cat(60)
@pytest.mark.parametrize('catalog,ids',[
    (cat(59),[f'w{i}' for i in range(60)]),(cat(61),[f'w{i}' for i in range(60)]),
    (cat(200),['w1','w0']),(cat(200),['w0','w0']),(cat(200),[]),
    (cat(200)[:-1]+[dict(window_id='w0')],['w0','w1'])])
def test_missing_order_and_duplicates(catalog,ids):
    with pytest.raises(ValueError):planned_catalog(catalog,ids)
