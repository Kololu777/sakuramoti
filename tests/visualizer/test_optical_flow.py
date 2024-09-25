from sakuramoti.visualizer.optical_flow import make_colorwheel


def test_make_colorwheel():
    colorwheel = make_colorwheel()
    assert colorwheel.shape == (55, 3)
