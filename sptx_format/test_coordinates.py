import pytest
from pkg_resources import resource_filename

from .util import SpaceTxValidator

package_name = "sptx_format"
coordinates_path = resource_filename(package_name, "schema/field_of_view/tiles/coordinates.json")
validator = SpaceTxValidator(coordinates_path)

example = resource_filename(package_name, "examples/field_of_view/coordinates_%s.json")


@pytest.mark.parametrize("name", (
    "bad_x_scalar",
    "bad_x_single",
    "bad_y_scalar",
    "bad_y_single",
    "bad_z_single",
    "bad_z_triple",
    "good_max",
    "good_min",
))
def test_coords(name):
    filename = example % name
    if "good" in name:
        assert validator.validate_file(filename)
    else:
        with pytest.warns(UserWarning):
            assert not validator.validate_file(filename)
