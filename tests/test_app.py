from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / 'app.py')


def test_model_first_and_optional_hardware():
    at = AppTest.from_file(APP, default_timeout=20).run()
    assert not at.exception
    assert any(m.label == 'Total inference memory' for m in at.metric)
    assert not any(m.label == 'Estimated maximum total context' for m in at.metric)
    at.number_input(key='params').set_value(14.0).run()
    at.checkbox(key='add_hardware').check().run()
    assert not at.exception
    assert at.number_input(key='params').value == 14.0
    assert any(m.label == 'Estimated maximum total context' for m in at.metric)
    assert len(at.get('download_button')) == 1


def test_sessions_independent():
    first = AppTest.from_file(APP, default_timeout=20).run()
    first.number_input(key='params').set_value(70.0).run()
    second = AppTest.from_file(APP, default_timeout=20).run()
    assert second.number_input(key='params').value == 7.0
    assert not second.exception


def test_unsupported_model_does_not_show_requirements():
    from calculator.hub import normalize
    from dataclasses import asdict
    at = AppTest.from_file(APP, default_timeout=20)
    at.session_state['metadata'] = asdict(normalize('test/moe', {'model_type': 'mixtral'}, {}))
    at.run()
    assert not at.exception
    assert not at.metric
    assert any('unsupported' in e.value.lower() for e in at.error)
