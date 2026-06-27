from rtw_llm.trl_compat import set_first_supported_kwarg, supported_config_kwargs


class DummyConfig:
    def __init__(self, keep: int = 0, max_length: int = 0):
        self.keep = keep
        self.max_length = max_length


def test_supported_config_kwargs_filters_unknown_keys():
    kwargs = supported_config_kwargs(DummyConfig, {"keep": 1, "drop": 2})
    assert kwargs == {"keep": 1}


def test_set_first_supported_kwarg_handles_renamed_keys():
    kwargs = {}
    set_first_supported_kwarg(DummyConfig, kwargs, ["max_seq_length", "max_length"], 128)
    assert kwargs == {"max_length": 128}
