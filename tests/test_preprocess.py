from src.data.preprocess import build_preprocessor


def test_build_preprocessor_returns_objects():
    preprocessor, num_features, cat_features = build_preprocessor()

    assert preprocessor is not None
    assert isinstance(num_features, list)
    assert isinstance(cat_features, list)
    assert "age" in num_features
    assert "gender" in cat_features
