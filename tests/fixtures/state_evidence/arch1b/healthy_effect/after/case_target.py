from target import normalize


def test_normalize() -> None:
    assert normalize(" PathLayer ") == "pathlayer"
