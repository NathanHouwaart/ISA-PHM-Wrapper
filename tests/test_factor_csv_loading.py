from isa_phm.factor_utils import load_factor_csv_dataframe, resolve_factor_csv_value


def test_loads_headered_and_unheadered_timeseries_and_scalar(tmp_path):
    (tmp_path / 'motor.csv').write_text('Time,Motor\n0.1,500\n0.2,510\n', encoding='utf-8')
    (tmp_path / 'numeric.csv').write_text('0.1,500\n0.2,510\n', encoding='utf-8')
    (tmp_path / 'scalar.csv').write_text('5.5\n', encoding='utf-8')
    frame = load_factor_csv_dataframe('motor.csv', tmp_path)
    assert list(frame.columns) == ['Time', 'Motor']
    assert frame['Motor'].tolist() == [500, 510]
    assert load_factor_csv_dataframe('numeric.csv', tmp_path).shape == (2, 2)
    assert resolve_factor_csv_value('motor.csv', tmp_path)['rows'] == 2
    assert resolve_factor_csv_value('scalar.csv', tmp_path) == 5.5
