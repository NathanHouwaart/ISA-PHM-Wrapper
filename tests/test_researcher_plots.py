import pandas as pd
import pytest

from isa_phm import ISAWrapper
from isa_phm.errors import DataFileError, PlotError
from isa_phm.schemas import FactorModel, ParameterValue


def test_plot_operating_conditions_uses_each_recorded_unit_and_reports_missing_file(minimal_single_run_isa_file, tmp_path):
    wrapper = ISAWrapper(minimal_single_run_isa_file, data_root=tmp_path, strict_validation=False)
    study = wrapper.study(1)
    run = study.assay(1).run('run_01')
    study._study.factors = [
        FactorModel(factor_id='load', factor_name='Axial load', factor_type='Operating condition', unit='N'),
        FactorModel(factor_id='speed', factor_name='Speed', factor_type='Operating condition', unit='rev/min'),
    ]
    run._run.factor_values = {'Axial load': './load.csv', 'Speed': './speed.csv'}
    (tmp_path / 'load.csv').write_text('Time,Load\n79,10\n80,20\n', encoding='utf8')
    (tmp_path / 'speed.csv').write_text('Time,Speed\n79.5,100\n80.5,200\n', encoding='utf8')
    layout = run.plot_operating_conditions()
    figures = [item[0] for item in layout.children]
    assert [plot.yaxis[0].axis_label for plot in figures] == ['Axial load (N)', 'Speed (rev/min)']
    combined = run.plot_with_operating_conditions(factor_names=['Axial load'])
    assert 'alignment is not verified' in combined.children[0].text
    sensor, condition = [item[0] for item in combined.children[1].children]
    assert sensor.x_range is condition.x_range
    assert condition.yaxis[0].axis_label == 'Axial load (N)'
    assert list(condition.renderers[0].data_source.data['x']) == [79, 80]
    aligned = run.plot_with_operating_conditions(alignment='first_sample')
    assert 'Actual synchronization is unknown' in aligned.children[0].text
    sensor, load, speed = [item[0] for item in aligned.children[1].children]
    assert sensor.renderers[0].data_source.data['t'][0] == 0
    assert list(load.renderers[0].data_source.data['x']) == [0, 1]
    assert list(speed.renderers[0].data_source.data['x']) == [0.5, 1.5]
    assert run.load_factor_timeseries('Axial load')['Time'].tolist() == [79, 80]
    with pytest.raises(ValueError, match='max_points'):
        run.plot_with_operating_conditions(max_points=0)
    (tmp_path / 'load.csv').unlink()
    with pytest.raises(DataFileError, match='Axial load'):
        run.plot_operating_conditions()


def test_sensor_feature_plot_checks_units_and_failed_runs(minimal_single_run_isa_file, tmp_path, monkeypatch):
    wrapper = ISAWrapper(minimal_single_run_isa_file, data_root=tmp_path, strict_validation=False)
    study = wrapper.study(1)
    sensor = study.assay(1)
    sensor._assay.runs[0].measurement_params = [ParameterValue(parameter_name='Measured Unit', value='fm')]
    frame = pd.DataFrame([{'run_id': 'run_01', 'run_number': 1, 'rms': 10.0, 'status': 'ok', 'error': ''}])
    monkeypatch.setattr(study, 'lifecycle_features', lambda *args, **kwargs: {'Sensor': frame})
    assert study.plot_sensor_features('rms').yaxis[0].axis_label == 'RMS (fm)'
    frame.loc[0, 'status'] = 'error'
    frame.loc[0, 'error'] = 'file missing'
    with pytest.raises(DataFileError, match='file missing'):
        study.plot_sensor_features('rms')
    sensor._assay.runs[0].measurement_params = []
    with pytest.raises(PlotError, match='unit unavailable'):
        study.plot_sensor_features('rms')
