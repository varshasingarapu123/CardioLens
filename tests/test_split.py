from preprocessing.split import split_records,assert_disjoint
def test_split_is_disjoint():
    ids=[str(i) for i in range(20)]
    train,val,test=split_records(ids,seed=42)
    assert_disjoint(train,val,test)
    assert set(train)|set(val)|set(test)==set(ids)
