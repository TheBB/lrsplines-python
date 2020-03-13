from pathlib import Path
import lrspline as lr


path = Path(__file__).parent


def test_read():
    with open(path / 'mesh01.lr', 'rb') as f:
        surf = lr.LRSplineSurface(f)

    assert len(surf.basis) == 1229
    assert len(surf.elements) == 1300
    assert len(surf.meshlines) == 130
