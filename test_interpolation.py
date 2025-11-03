import numpy as np
import pytest

from interpolation import NotFittedError, Splines


@pytest.fixture
def simple_data():
    """Provides simple data for testing."""
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([0, 1, 4, 9, 16])  # y = x^2
    return x, y


@pytest.fixture
def fitted_spline(simple_data):
    """Provides a fitted Splines instance."""
    x, y = simple_data
    spline = Splines()
    spline.fit(x, y)
    return spline


def test_splines_init():
    """Tests the initial state of the Splines instance."""
    spline = Splines()
    assert spline.x_ is None
    assert spline.y_ is None
    assert spline.coeffs_ is None
    assert spline._n_splines == 0


def test_fit_unequal_dims():
    """Tests that fit() raises ValueError for unequal input dimensions."""
    spline = Splines()
    x = np.array([1, 2, 3])
    y = np.array([1, 2])
    with pytest.raises(ValueError, match="x and y must have the same dimensions"):
        spline.fit(x, y)


def test_fit_sets_attributes(simple_data):
    """Tests that fit() correctly sets the instance attributes."""
    x, y = simple_data
    # Unsorted data to test sorting
    x_unsorted = np.array([2, 0, 4, 1, 3])
    y_unsorted = np.array([4, 0, 16, 1, 9])

    spline = Splines()
    spline.fit(x_unsorted, y_unsorted)

    n = len(x)
    assert spline._n_splines == n - 1
    np.testing.assert_array_equal(spline.x_, x)
    np.testing.assert_array_equal(spline.y_, y)
    assert spline.coeffs_ is not None
    assert spline.coeffs_.shape == (4, n - 1)


def test_predict_before_fit():
    """Tests that predict() raises NotFittedError if called before fit()."""
    spline = Splines()
    with pytest.raises(NotFittedError):
        spline.predict(1.0)


def test_predict_outside_range(fitted_spline):
    """Tests that predict() raises ValueError for x outside the fitted range."""
    with pytest.raises(
        ValueError, match=r"x values must be within the fitted range \[0.0, 4.0\]."
    ):
        fitted_spline.predict(-1.0)

    with pytest.raises(
        ValueError, match=r"x values must be within the fitted range \[0.0, 4.0\]."
    ):
        fitted_spline.predict(5.0)

    with pytest.raises(
        ValueError, match=r"x values must be within the fitted range \[0.0, 4.0\]."
    ):
        fitted_spline.predict(np.array([1.0, 5.0]))


def test_predict_at_knots(fitted_spline, simple_data):
    """Tests that predictions at the original data points (knots) are exact."""
    x, y = simple_data
    y_pred = fitted_spline.predict(x)
    np.testing.assert_allclose(y_pred, y, atol=1e-9)


def test_predict_scalar_and_array(fitted_spline):
    """Tests prediction with both scalar and array inputs."""
    # Test scalar input
    y_pred_scalar = fitted_spline.predict(1.5)
    assert isinstance(y_pred_scalar, float)
    assert np.isclose(
        y_pred_scalar, 1.5**2, atol=0.1
    )  # y=x^2, natural spline is an approximation

    # Test array input
    x_pred = np.array([0.5, 1.5, 2.5, 3.5])
    y_pred_array = fitted_spline.predict(x_pred)
    assert isinstance(y_pred_array, np.ndarray)
    assert y_pred_array.shape == x_pred.shape

    # The prediction for 1.5 should be the same in both cases
    assert np.isclose(y_pred_scalar, y_pred_array[1])


def test_interpolation_accuracy_linear():
    """Tests the interpolation accuracy on a simple linear function."""
    x = np.linspace(0, 10, 11)
    y = 2 * x + 5  # y = 2x + 5

    spline = Splines()
    spline.fit(x, y)

    x_test = np.array([1.5, 3.3, 7.8, 9.1])
    y_true = 2 * x_test + 5
    y_pred = spline.predict(x_test)

    # For a linear function, natural cubic splines should be exact.
    np.testing.assert_allclose(y_pred, y_true, atol=1e-9)


def test_predict_on_boundaries(fitted_spline, simple_data):
    """Tests prediction at the exact boundaries of the interval."""
    x, y = simple_data
    y_pred_start = fitted_spline.predict(x[0])
    y_pred_end = fitted_spline.predict(x[-1])
    assert np.isclose(y_pred_start, y[0])
    assert np.isclose(y_pred_end, y[-1])
