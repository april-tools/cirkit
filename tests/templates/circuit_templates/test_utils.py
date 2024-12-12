from cirkit.templates.utils import Parameterization, parameterization_to_factory


def test_param_activation_kwargs():
    min_var = 0.1
    max_var = 1.0
    stddev_param = Parameterization(
        initialization="normal",
        activation="positive-clamp",
        activation_kwargs={"vmin": min_var, "vmax": max_var},
    )
    stddev_param_factory = parameterization_to_factory(stddev_param)
    parameter = stddev_param_factory(
        shape=(1, 1),
    )
    assert (
        parameter.output.vmin == min_var
    ), f"parameter min value expected: {min_var}, got: {parameter.output.vmin}"
    assert (
        parameter.output.vmax == max_var
    ), f"parameter ma value expected: {max_var}, got: {parameter.output.vmax}"
