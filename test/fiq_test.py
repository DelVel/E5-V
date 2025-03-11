import PIL

from src.datasets.fashion_iq import get_fiq_image_dataset, get_fiq_label_dataset


def test_column_names():
    dset = get_fiq_label_dataset("train", ["dress"])
    assert set(dset.column_names) == {'iq_id','iq', 'tq', 'it', 'it_id'}

def test_query_columns():
    dset = get_fiq_label_dataset("train", ["dress"])
    elem0 = dset[0]
    assert isinstance(elem0['iq'], PIL.Image.Image)
    assert isinstance(elem0['it'], PIL.Image.Image)
    assert isinstance(elem0['iq_id'], str)
    assert isinstance(elem0['it_id'], str)
    assert isinstance(elem0['tq'], list)
    assert isinstance(elem0['tq'][0], str)

def test_img_dset():
    dset = get_fiq_image_dataset("train", ["dress"])
    elem0 = dset[0]
    assert isinstance(elem0['image'], PIL.Image.Image)
    assert isinstance(elem0['id'], str)
