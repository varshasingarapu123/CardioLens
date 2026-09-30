from preprocessing.labels import map_symbol,class_id
def test_normal_mapping():
    assert map_symbol('N')=='N' and class_id('N')==0
def test_ventricular_mapping():
    assert map_symbol('V')=='V' and class_id('V')==2
def test_unknown_symbol_is_skipped():
    assert map_symbol('not-a-beat') is None
