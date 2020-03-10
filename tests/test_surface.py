from pathlib import Path
import lrspline as lr


path = Path(__file__).parent


def test_read():
    with open(path / 'mesh01.lr', 'rb') as f:
        surf = lr.LRSplineSurface(f)

    assert len(list(surf.basis())) == 1229
    assert len(list(surf.elements())) == 1300
    assert len(list(surf.meshlines())) == 130
