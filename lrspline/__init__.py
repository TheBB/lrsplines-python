from __future__ import annotations

import io
import operator as op
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Sequence
from functools import partial
from itertools import chain, combinations_with_replacement, repeat
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Generic,
    Literal,
    Protocol,
    Self,
    SupportsFloat,
    TextIO,
    TypedDict,
    TypeVar,
    Unpack,
    cast,
    overload,
)

import numpy as np
from numpy.typing import NDArray

from . import raw

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

__version__ = "1.14.4"

L = TypeVar("L", bound="LRSplineObject")
R = TypeVar("R", bound=raw.LRSplineObject)
S = TypeVar("S")
T = TypeVar("T")
U = TypeVar("U")

Direction = int | Literal["u", "U", "v", "V", "w", "W"]
Edge = Literal["west", "east", "north", "south", "top", "bottom"]
FArray = NDArray[np.floating]
Scalar = float | np.floating
Scalars = Sequence[Scalar] | FArray
ScalarOrScalars = Scalar | Scalars


class ConfigureKwargs(TypedDict, total=False):
    aspect_ratio: float
    posteriori_fix: bool
    strategy: int
    symmetry: int
    multiplicity: int
    max_tjoints: int
    close_gaps: bool


class WritePsKwargs(TypedDict, total=False):
    nu: int
    nv: int
    close: bool
    colorElements: Sequence[int] | None


def _ensure_scalars(x: ScalarOrScalars | tuple[Scalar], dups: int = 1) -> list[float]:
    if isinstance(x, SupportsFloat) and not isinstance(x, np.ndarray):
        return [float(x)] * dups
    retval = list(map(float, x))
    if len(retval) < dups and retval:
        retval.extend(float(x[-1]) for _ in range(dups - len(retval)))
    return retval


def _check_direction(direction: Direction, pardim: int) -> int:
    if direction in {0, "u", "U"} and pardim > 0:
        return 0
    if direction in {1, "v", "V"} and pardim > 1:
        return 1
    if direction in {2, "w", "W"} and pardim > 2:
        return 2
    raise ValueError("Invalid direction")


def _check_edge(edge: Iterable[Edge]) -> int:
    side = raw.parameterEdge.NONE
    for arg in edge:
        side |= {
            "west": raw.parameterEdge.WEST,
            "east": raw.parameterEdge.EAST,
            "south": raw.parameterEdge.SOUTH,
            "north": raw.parameterEdge.NORTH,
            "top": raw.parameterEdge.TOP,
            "bottom": raw.parameterEdge.BOTTOM,
        }[arg]
    return side


def _constructor(stream: TextIO | BinaryIO | bytes | str) -> type[LRSplineObject]:
    if isinstance(stream, TextIO | BinaryIO):
        # if hasattr(stream, "readline"):
        peek = stream.readline()
        if not peek:
            raise raw.EOFError("")
    else:
        peek = stream
    if isinstance(peek, bytes):
        peek = peek.decode("utf-8")
    if peek.startswith("# LRSPLINE SURFACE"):
        return LRSplineSurface
    if peek.startswith("# LRSPLINE VOLUME"):
        return LRSplineVolume
    raise ValueError(f"Unknown LRSpline object type: '{peek}'")


def _derivative_index(d: tuple[int, int] | tuple[int, int, int]) -> tuple[int, int]:
    """Calculate the derivative index of 'd' (a 2-tuple or 3-tuple) using
    LRSplines' derivative numbering scheme.  Return nderivs and index.
    """
    nderivs = sum(d)
    if len(d) == 2:
        index = sum(dd + 1 for dd in range(nderivs)) + d[1]
        return nderivs, index
    index = nderivs * (nderivs + 1) * (nderivs + 2) // 6
    tgt = tuple(chain.from_iterable(repeat(i, r) for i, r in enumerate(d)))
    index += next(i for i, t in enumerate(combinations_with_replacement(range(len(d)), nderivs)) if t == tgt)
    return nderivs, index


class DerivativeHelperFunc2d(Protocol):
    def __call__(self, u: np.floating, v: np.floating, nderivs: int) -> FArray: ...


class DerivativeHelperFunc3d(Protocol):
    def __call__(self, u: np.floating, v: np.floating, w: np.floating, nderivs: int) -> FArray: ...


@overload
def _derivative_helper(
    pts: tuple[FArray, FArray] | tuple[Scalar, Scalar],
    derivs: tuple[int, int] | Sequence[tuple[int, int]],
    func: DerivativeHelperFunc2d,
) -> FArray: ...


@overload
def _derivative_helper(
    pts: tuple[FArray, FArray, FArray] | tuple[Scalar, Scalar, Scalar],
    derivs: tuple[int, int, int] | Sequence[tuple[int, int, int]],
    func: DerivativeHelperFunc3d,
) -> FArray: ...


def _derivative_helper(pts, derivs, func):  # type: ignore[no-untyped-def]
    """Helper for calculating derivatives.

    Pts must be a tuple of points, or a tuple of point arrays.

    Derivs must be a tuple indicating which derivative to take, or a
    list of such tuples.

    Func must be a callable accepting two or three floats (the point
    to evaluate) and an 'nderivs' argument, returning a 1-dimensional
    array.

    - If one point, one deriv: return a 1-dimensional array (coord)
    - If multiple points, one deriv: return a 2-dimensional array (point x coord)
    - If one point, multiple derivs: return a 2-dimensional array (deriv x coord)
    - If multiple points, multiple derivs: return 3-dimensional (deriv x point x coord)
    """

    singlept = not isinstance(pts[0], np.ndarray)

    # If requesting a single derivative
    if isinstance(derivs[0], int):
        nderiv, index = _derivative_index(derivs)

        # If requesting a single point
        if singlept:
            # pts = cast(tuple[Scalar, ...], pts)
            return cast("FArray", func(*pts, nderivs=nderiv)[index])

        # If requesting multiple points
        data_list = [func(*pt, nderivs=nderiv) for pt in zip(*pts)]
        return np.array([d[index] for d in data_list])

    # If requesting several derivatives
    nderiv, indexes = 0, []
    for deriv in derivs:
        n, i = _derivative_index(deriv)
        nderiv = max(n, nderiv)
        indexes.append(i)

    # If requesting a single point
    if singlept:
        # pts = cast(tuple[Scalar, ...], pts)
        data = func(*pts, nderivs=nderiv)
        return np.array([data[i] for i in indexes])

    # If requesting multiple points
    data_list = [func(*pt, nderivs=nderiv) for pt in zip(*pts)]
    return np.array([[d[i] for d in data_list] for i in indexes])


class SimpleWrapper(Generic[T]):
    lr: LRSplineObject
    w: T

    def __init__(self, lr: LRSplineObject, w: T) -> None:
        self.w = w
        self.lr = lr


class BasisFunction(SimpleWrapper[raw.Basisfunction]):
    def __str__(self) -> str:
        return f"BasisFunction#{self.id}"

    @property
    def id(self) -> int:
        return self.w.getId()

    @property
    def controlpoint(self) -> FArray:
        return self.w.getControlPoint()

    @property
    def nvariate(self) -> int:
        return self.w.nVariate()

    def evaluate(self, u: FArray, v: FArray) -> FArray:
        retval = np.array([self.w.evaluate(up, vp, True, True) for up, vp in zip(u.flatten(), v.flatten())])
        return retval.reshape(u.shape)

    def derivative(self, *pts: Scalar | FArray, d: tuple[int, int] | tuple[int, int, int] = (1, 1)) -> FArray:
        if self.nvariate == 2:

            def wrapper_2d(u: np.floating, v: np.floating, nderivs: int) -> FArray:
                return self.w.evaluate(u, v, nderivs, True, True)

            pts = cast("tuple[Scalar, Scalar] | tuple[FArray, FArray]", pts)
            d = cast("tuple[int, int]", d)
            return _derivative_helper(pts, d, wrapper_2d)

        def wrapper_3d(u: np.floating, v: np.floating, w: np.floating, nderivs: int) -> FArray:
            return self.w.evaluate(u, v, w, nderivs, True, True, True)

        pts = cast("tuple[Scalar, Scalar, Scalar] | tuple[FArray, FArray, FArray]", pts)
        d = cast("tuple[int, int, int]", d)
        return _derivative_helper(pts, d, wrapper_3d)

    def support(self) -> Iterator[Element]:
        for w in self.w.supportIter():
            yield Element(self.lr, w)

    def __getitem__(self, idx: int) -> FArray:
        return self.w.getknots(idx)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BasisFunction):
            return False
        return self.id == other.id

    __call__ = evaluate

    def refine(self) -> None:
        self.lr.w.refineBasisFunction(self.id)
        self.lr.w.generateIDs()


class Element(SimpleWrapper[raw.Element]):
    def __str__(self) -> str:
        return f"Element#{self.id}"

    @property
    def id(self) -> int:
        return self.w.getId()

    @property
    def pardim(self) -> int:
        return self.w.getDim()

    @overload
    def start(self) -> tuple[float, ...]: ...

    @overload
    def start(self, direction: Direction) -> float: ...

    def start(self, direction=None):  # type: ignore[no-untyped-def]
        if direction is None:
            return tuple(self.w.getParmin(d) for d in range(self.pardim))
        return self.w.getParmin(_check_direction(direction, self.pardim))

    @overload
    def end(self) -> tuple[float, ...]: ...

    @overload
    def end(self, direction: Direction) -> float: ...

    def end(self, direction=None):  # type: ignore[no-untyped-def]
        if direction is None:
            return tuple(self.w.getParmax(d) for d in range(self.pardim))
        return self.w.getParmax(_check_direction(direction, self.pardim))

    @overload
    def span(self) -> tuple[tuple[float, float], ...]: ...

    @overload
    def span(self, direction: Direction) -> tuple[float, float]: ...

    def span(self, direction=None):  # type: ignore[no-untyped-def]
        if direction is None:
            return tuple((self.w.getParmin(d), self.w.getParmax(d)) for d in range(self.pardim))
        direction = _check_direction(direction, self.pardim)
        return (self.w.getParmin(direction), self.w.getParmax(direction))

    def support(self) -> Iterator[BasisFunction]:
        for w in self.w.supportIter():
            yield BasisFunction(self.lr, w)

    def refine(self) -> None:
        self.lr.w.refineElement(self.id)
        self.lr.w.generateIDs()

    def bezier_extraction(self) -> FArray:
        return self.lr.bezier_extraction(self.id)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Element):
            return False
        return self.id == other.id


I = TypeVar("I", raw.Meshline, raw.MeshRectangle)


class MeshInterface(ABC, SimpleWrapper[I]):
    @property
    @abstractmethod
    def constant_direction(self) -> int: ...

    @property
    @abstractmethod
    def value(self) -> float: ...

    @property
    @abstractmethod
    def variable_directions(self) -> tuple[int, ...]: ...

    @property
    @abstractmethod
    def multiplicity(self) -> int: ...

    @overload
    def span(self) -> tuple[tuple[float, float], ...]: ...

    @overload
    def span(self, direction: Direction) -> tuple[float, float]: ...

    @abstractmethod
    def span(self, direction=None):  # type: ignore[no-untyped-def]
        ...

    def __str__(self) -> str:
        cls = self.__class__.__name__
        const = "{} = {}".format("uvw"[self.constant_direction], self.value)
        variables = [
            "{} < {} < {}".format(self.start(i), "uvw"[i], self.end(i)) for i in self.variable_directions
        ]
        return cls + "(" + const + "; " + ", ".join(variables) + ")"

    @overload
    def start(self) -> tuple[float, ...]: ...

    @overload
    def start(self, direction: Direction) -> float: ...

    def start(self, direction=None):  # type: ignore[no-untyped-def]
        if direction is None:
            return tuple(self.span(d)[0] for d in range(self.lr.pardim))
        return self.span(direction)[0]

    @overload
    def end(self) -> tuple[float, ...]: ...

    @overload
    def end(self, direction: Direction) -> float: ...

    def end(self, direction=None):  # type: ignore[no-untyped-def]
        if direction is None:
            return tuple(self.span(d)[1] for d in range(self.lr.pardim))
        return self.span(direction)[1]


class MeshLine(MeshInterface[raw.Meshline]):
    @property
    def variable_direction(self) -> int:
        return 0 if self.w.is_spanning_u() else 1

    @property
    def variable_directions(self) -> tuple[int, ...]:
        return (0,) if self.w.is_spanning_u() else (1,)

    @property
    def constant_direction(self) -> int:
        return 1 if self.w.is_spanning_u() else 0

    @property
    def value(self) -> float:
        return self.w.const_par_

    @property
    def multiplicity(self) -> int:
        return self.w.multiplicity_

    def span(self, direction=None):  # type: ignore[no-untyped-def]
        if direction is None:
            var = (self.w.start_, self.w.stop_)
            const = (self.w.const_par_, self.w.const_par_)
            return (var, const) if self.w.is_spanning_u() else (const, var)
        direction = _check_direction(direction, self.lr.pardim)
        if direction == self.variable_direction:
            return (self.w.start_, self.w.stop_)
        return (self.w.const_par_, self.w.const_par_)


class MeshRect(MeshInterface[raw.MeshRectangle]):
    @property
    def variable_directions(self) -> tuple[int, ...]:
        return {
            0: (1, 2),
            1: (0, 2),
            2: (0, 1),
        }[self.w.constDirection()]

    @property
    def constant_direction(self) -> int:
        return self.w.constDirection()

    @property
    def value(self) -> float:
        return self.w.constParameter()

    @property
    def multiplicity(self) -> int:
        return self.w.multiplicity_

    def span(self, direction=None):  # type: ignore[no-untyped-def]
        start, end = self.w.start_, self.w.stop_
        if direction is None:
            return tuple(zip(start, end))
        direction = _check_direction(direction, self.lr.pardim)
        return start[direction], end[direction]


class ListLikeView(Generic[S, T, U]):
    obj: S
    lenf: Callable[[S], int]
    itemf: Callable[[S], Callable[[int], T]]
    iterf: Callable[[S], Iterator[T]]
    wrapf: Callable[[T], U]

    def __init__(self, obj: S, lenf: str, itemf: str, iterf: str, wrapf: Callable[[T], U]):
        self.obj = obj
        self.lenf = cast("Callable[[S], int]", op.methodcaller(lenf))
        self.itemf = cast("Callable[[S], Callable[[int], T]]", op.attrgetter(itemf))
        self.iterf = cast("Callable[[S], Iterator[T]]", op.methodcaller(iterf))
        self.wrapf = wrapf

    def __len__(self) -> int:
        return self.lenf(self.obj)

    def __getitem__(self, idx: int) -> U:
        return self.wrapf(self.itemf(self.obj)(idx))

    def __iter__(self) -> Iterator[U]:
        for w in self.iterf(self.obj):
            yield self.wrapf(w)


class ElementView(ListLikeView[R, raw.Element, Element]):
    lr: LRSplineObject[R]

    def __init__(self, lr: LRSplineObject[R]) -> None:
        super().__init__(lr.w, "nElements", "getElement", "elementIter", partial(Element, lr))
        self.lr = lr

    def edge(self, *edge: Edge) -> Iterator[Element]:
        for w in self.lr.w.getEdgeElementsIter(_check_edge(edge)):
            yield Element(self.lr, w)


class BasisView(ListLikeView[R, raw.Basisfunction, BasisFunction]):
    lr: LRSplineObject[R]

    def __init__(self, lr: LRSplineObject[R]) -> None:
        super().__init__(lr.w, "nBasisFunctions", "getBasisfunction", "basisIter", partial(BasisFunction, lr))
        self.lr = lr

    def edge(self, *edge: Edge, depth: int = 1) -> Iterator[BasisFunction]:
        for w in self.lr.w.getEdgeFunctionsIter(_check_edge(edge)):
            yield BasisFunction(self.lr, w)


class MeshLineView(ListLikeView[raw.LRSurface, raw.Meshline, MeshLine]):
    def __init__(self, lr: LRSplineSurface) -> None:
        super().__init__(lr.w, "nMeshlines", "getMeshline", "meshlineIter", partial(MeshLine, lr))


class MeshRectView(ListLikeView[raw.LRVolume, raw.MeshRectangle, MeshRect]):
    def __init__(self, lr: LRSplineVolume) -> None:
        super().__init__(lr.w, "nMeshRectangles", "getMeshRectangle", "meshrectIter", partial(MeshRect, lr))


class LRSplineObject(ABC, Generic[R]):
    w: R
    elements: ElementView[R]
    basis: BasisView[R]

    @abstractmethod
    def clone(self) -> Self:
        pass

    def __init__(self, w: R, renumber: bool = True) -> None:
        if renumber:
            w.generateIDs()
        self.w = w
        self.elements = ElementView(self)
        self.basis = BasisView(self)

    @staticmethod
    def read_many(stream: BinaryIO | TextIO, renumber: bool = True) -> list[LRSplineObject]:
        contents = stream.read()
        splitter = b"# LRSPLINE" if isinstance(contents, bytes) else "# LRSPLINE"
        contents_list = contents.split(splitter)[1:]  # type: ignore[arg-type]
        contents_list = [splitter + c for c in contents_list]  # type: ignore[operator,assignment]
        return [_constructor(c)(c, renumber=renumber) for c in contents_list]

    @abstractmethod
    def corners(self) -> FArray: ...

    def __len__(self) -> int:
        return len(self.basis)

    @property
    def pardim(self) -> int:
        return next(iter(self.elements)).pardim

    @property
    def dimension(self) -> int:
        return len(next(iter(self.basis)).controlpoint)

    @dimension.setter
    def dimension(self, value: int) -> None:
        self.w.rebuildDimension(value)

    @property
    def shape(self) -> tuple[int]:
        return (len(self),)

    @property
    def controlpoints(self) -> FArray:
        return np.array([bf.controlpoint for bf in self.basis])

    @controlpoints.setter
    def controlpoints(self, value: FArray) -> None:
        if value.shape[0] != len(self):
            raise ValueError(f"Incorrect number of control points: expected {len(self)}")
        if value.ndim != 2:
            raise ValueError("Incorrect number of dimensions: expected 2")
        _, newdim = value.shape
        if newdim != self.dimension:
            self.dimension = newdim
        self.w.setControlpoints(value.flatten())

    def write(self, stream: TextIO) -> None:
        return self.w.write(stream)

    @overload
    def start(self) -> tuple[float, ...]: ...

    @overload
    def start(self, direction: Direction) -> float: ...

    def start(self, direction=None):  # type: ignore[no-untyped-def]
        if direction is None:
            return tuple(self.w.startparam(d) for d in range(self.pardim))
        return self.w.startparam(_check_direction(direction, self.pardim))

    @overload
    def end(self) -> tuple[float, ...]: ...

    @overload
    def end(self, direction: Direction) -> float: ...

    def end(self, direction=None):  # type: ignore[no-untyped-def]
        if direction is None:
            return tuple(self.w.endparam(d) for d in range(self.pardim))
        return self.w.endparam(_check_direction(direction, self.pardim))

    @overload
    def span(self) -> tuple[tuple[float, float], ...]: ...

    @overload
    def span(self, direction: Direction) -> tuple[float, float]: ...

    def span(self, direction=None):  # type: ignore[no-untyped-def]
        if direction is None:
            return tuple((self.w.startparam(d), self.w.endparam(d)) for d in range(self.pardim))
        direction = _check_direction(direction, self.pardim)
        return (self.w.startparam(direction), self.w.endparam(direction))

    @overload
    def order(self) -> tuple[int, ...]: ...

    @overload
    def order(self, direction: Direction) -> int: ...

    def order(self, direction=None):  # type: ignore[no-untyped-def]
        if direction is None:
            return tuple(self.w.order(d) for d in range(self.pardim))
        return self.w.order(_check_direction(direction, self.pardim))

    @overload
    def knots(self, *, with_multiplicities: bool = ...) -> tuple[FArray, ...]: ...

    @overload
    def knots(self, direction: Direction, *, with_multiplicities: bool = ...) -> FArray: ...

    def knots(self, direction=None, with_multiplicities=False):  # type: ignore[no-untyped-def]
        knots = self.w.getGlobalKnotVector() if with_multiplicities else self.w.getGlobalUniqueKnotVector()

        if direction is None:
            return knots
        direction = _check_direction(direction, self.pardim)
        return knots[direction]

    def refine(
        self,
        objects: Sequence[float] | Sequence[BasisFunction] | Sequence[Element],
        beta: float | None = None,
    ) -> None:
        if not objects:
            raise ValueError("Refinement list must be non-empty")
        if isinstance(objects[0], float):
            assert isinstance(beta, float)
            self.w.refineByDimensionIncrease(np.asarray(objects), beta)
        elif isinstance(objects[0], BasisFunction):
            objects = cast("Sequence[BasisFunction]", objects)
            ids = [bf.id for bf in objects]
            self.w.refineBasisFunction(ids)
        elif isinstance(objects[0], Element):
            objects = cast("Sequence[Element]", objects)
            ids = [bf.id for bf in objects]
            self.w.refineElement(ids)
        else:
            raise TypeError("List of unknown objects: expected float, BasisFunction or Element")
        self.w.generateIDs()

    def configure(self, **kwargs: Unpack[ConfigureKwargs]) -> None:
        if "aspect_ratio" in kwargs:
            r = kwargs.pop("aspect_ratio")
            posteriori = kwargs.pop("posteriori_fix", True)
            self.w.setMaxAspectRatio(r, posteriori)
        for key, val in kwargs.items():
            {
                "strategy": "setRefStrat",
                "symmetry": "setRefSymmetry",
                "multiplicity": "setRefMultiplicity",
                "max_tjoints": "setMaxTjoints",
                "close_gaps": "setCloseGaps",
            }[key]
            getattr(self.w, key)(val)

    def generate_ids(self) -> None:
        self.w.generateIDs()

    def bezier_extraction(self, arg: Element | int) -> FArray:
        if isinstance(arg, Element):
            return self.w.getBezierExtraction(arg.id)
        return self.w.getBezierExtraction(arg)  # int

    def __mul__(self, x: Any) -> Self:
        new = self.clone()
        new.controlpoints *= x
        return new

    def __rmul__(self, x: Any) -> Self:
        return self.__mul__(x)

    def __getitem__(self, i: int) -> FArray:
        return self.basis[i].controlpoint

    @abstractmethod
    def element_at(self, *args: float) -> Element: ...

    @abstractmethod
    def __call__(self, *args: FArray | float, iel: int = ...) -> FArray: ...


class LRSplineSurface(LRSplineObject[raw.LRSurface]):
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, surface: raw.LRSurface) -> None: ...

    @overload
    def __init__(self, data: BinaryIO | TextIO | str | bytes) -> None: ...

    @overload
    def __init__(self, n1: int, n2: int) -> None: ...

    @overload
    def __init__(self, n1: int, n2: int, order_u: int, order_v: int) -> None: ...

    @overload
    def __init__(
        self, n1: int, n2: int, order_u: int, order_v: int, knot1: Scalars, knot2: Scalars
    ) -> None: ...

    @overload
    def __init__(
        self, n1: int, n2: int, order_u: int, order_v: int, knot1: Scalars, knot2: Scalars, cps: ArrayLike
    ) -> None: ...

    def __init__(self, *args, renumber=True):  # type: ignore[no-untyped-def]
        if len(args) == 0:
            w = raw.LRSurface()
        elif isinstance(args[0], raw.LRSurface):
            w = args[0]
        elif isinstance(args[0], io.IOBase | str | bytes):
            w = raw.LRSurface()
            w.read(args[0])  # type: ignore[arg-type]
        elif len(args) == 2:  # specify (n1,n2)
            w = raw.LRSurface(args[0], args[1], 2, 2)
        elif len(args) == 4:  # specify (n1,n2) and (p1,p2)
            w = raw.LRSurface(args[0], args[1], args[2], args[3])
        elif len(args) == 6:  # specify (n1,n2), (p1,p2) and (knot1,knot2)
            w = raw.LRSurface(args[0], args[1], args[2], args[3], args[4], args[5])
        elif len(args) == 7:  # specify controlpoints in addition
            cp = np.asarray(args[-1], dtype=float)
            w = raw.LRSurface(args[0], args[1], args[2], args[3], args[4], args[5], cp.flat, len(cp[0]))
        else:
            w = raw.LRSurface()
        super().__init__(w, renumber=renumber)
        self.meshlines = MeshLineView(self)

    def corners(self) -> FArray:
        return np.array(
            [
                next(self.basis.edge("east", "south")).controlpoint,
                next(self.basis.edge("west", "south")).controlpoint,
                next(self.basis.edge("east", "north")).controlpoint,
                next(self.basis.edge("west", "north")).controlpoint,
            ]
        )

    def clone(self) -> LRSplineSurface:
        return LRSplineSurface(self.w.copy())

    def write_postscript(self, stream: TextIO, **kwargs: Unpack[WritePsKwargs]) -> None:
        return self.w.writePostscriptElements(stream, **kwargs)

    def insert_knot(self, new_knots: ScalarOrScalars, direction: Direction, multiplicity: int = 1) -> None:
        new_knots = _ensure_scalars(new_knots)
        direction = _check_direction(direction, self.pardim)
        for k in new_knots:
            if direction == 0:
                self.w.insert_const_u_edge(k, self.start(1), self.end(1), multiplicity)
            else:
                self.w.insert_const_v_edge(k, self.start(0), self.end(0), multiplicity)
        self.w.generateIDs()

    @overload
    def insert(
        self,
        arg: MeshLine,
        /,
        direction: Direction | None = ...,
        value: float | None = ...,
        start: float | None = ...,
        end: float | None = ...,
        multiplicity: int | None = ...,
    ) -> None: ...

    @overload
    def insert(
        self,
        direction: Direction,
        value: float,
        start: float,
        end: float,
        multiplicity: int | None = ...,
    ) -> None: ...

    def insert(self, *args, direction=None, value=None, start=None, end=None, multiplicity=None):  # type: ignore[no-untyped-def]
        if len(args) > 1:
            raise TypeError("Expected at most one positional argument")

        if len(args) == 1:
            (ml,) = args
            if not isinstance(ml, MeshLine):
                raise TypeError(f"Expected MeshLine, got {type(ml)}")
            if direction is None:
                direction = ml.constant_direction
            if value is None:
                value = ml.value
            if start is None:
                start = ml.start(ml.variable_direction)
            if end is None:
                end = ml.end(ml.variable_direction)
            if multiplicity is None:
                multiplicity = ml.multiplicity

        assert direction is not None
        assert value is not None
        assert start is not None
        assert end is not None

        if multiplicity is None:
            multiplicity = 1
        direction = _check_direction(direction, self.pardim)

        if direction == 0:
            self.w.insert_const_u_edge(value, start, end, multiplicity)
        else:
            self.w.insert_const_v_edge(value, start, end, multiplicity)

    @overload
    def evaluate(self, u: FArray, v: FArray, iel: int = ...) -> FArray: ...

    @overload
    def evaluate(self, u: float, v: float, iel: int = ...) -> FArray: ...

    def evaluate(self, u, v, iel=-1):  # type: ignore[no-untyped-def]
        if isinstance(u, np.ndarray) and isinstance(v, np.ndarray):
            retval = np.array([self.w.point(up, vp, iEl=iel) for up, vp in zip(u.flat, v.flat)])
            return retval.reshape(u.shape + (-1,))
        return self.w.point(u, v, iEl=iel)

    @overload
    def derivative(self, u: FArray, v: FArray, d: tuple[int, int] = ..., iel: int = ...) -> FArray: ...

    @overload
    def derivative(self, u: float, v: float, d: tuple[int, int] = ..., iel: int = ...) -> FArray: ...

    def derivative(self, u, v, d=(1, 1), iel=-1):  # type: ignore[no-untyped-def]
        def wrapper(u: np.floating, v: np.floating, nderivs: int) -> FArray:
            return self.w.point(u, v, nderivs, iEl=iel)

        return _derivative_helper((u, v), d, wrapper)

    def element_at(self, u: float, v: float) -> Element:  # type: ignore[override]
        return self.elements[self.w.getElementContaining(u, v)]

    def make_integer_knots(self) -> float:
        return self.w.makeIntegerKnots()

    __call__ = evaluate  # type: ignore[assignment]


class LRSplineVolume(LRSplineObject[raw.LRVolume]):
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, surface: raw.LRVolume) -> None: ...

    @overload
    def __init__(self, data: BinaryIO | TextIO | str | bytes) -> None: ...

    @overload
    def __init__(self, n1: int, n2: int, n3: int) -> None: ...

    @overload
    def __init__(self, n1: int, n2: int, n3: int, order_u: int, order_v: int, order_w: int) -> None: ...

    @overload
    def __init__(
        self,
        n1: int,
        n2: int,
        n3: int,
        order_u: int,
        order_v: int,
        order_w: int,
        knot1: Scalars,
        knot2: Scalars,
        knot3: Scalars,
    ) -> None: ...

    @overload
    def __init__(
        self,
        n1: int,
        n2: int,
        n3: int,
        order_u: int,
        order_v: int,
        order_w: int,
        knot1: Scalars,
        knot2: Scalars,
        knot3: Scalars,
        cps: ArrayLike,
    ) -> None: ...

    def __init__(self, *args, renumber=True):  # type: ignore[no-untyped-def]
        if len(args) == 0:
            w = raw.LRVolume()
        elif isinstance(args[0], raw.LRVolume):
            w = args[0]
        elif isinstance(args[0], io.IOBase | str | bytes):
            w = raw.LRVolume()
            w.read(args[0])  # type: ignore[arg-type]
        elif len(args) == 3:  # only specify number of functions (n1,n2,n3)
            w = raw.LRVolume(args[0], args[1], args[2], 2, 2, 2)
        elif len(args) == 6 or len(args) == 9:  # specity n & p for 3 directions
            w = raw.LRVolume(*args)
        elif len(args) == 10:  # specify all above in addition to controlpoints
            cp = np.array(args[-1])
            w = raw.LRVolume(*args[:-1], cp.flatten(), len(cp[0]))  # type: ignore[call-overload]
        else:
            w = raw.LRVolume()
        super().__init__(w, renumber=renumber)
        self.meshrects = MeshRectView(self)

    def corners(self) -> FArray:
        return np.array(
            [
                next(self.basis.edge("east", "south", "bottom")).controlpoint,
                next(self.basis.edge("west", "south", "bottom")).controlpoint,
                next(self.basis.edge("east", "north", "bottom")).controlpoint,
                next(self.basis.edge("west", "north", "bottom")).controlpoint,
                next(self.basis.edge("east", "south", "top")).controlpoint,
                next(self.basis.edge("west", "south", "top")).controlpoint,
                next(self.basis.edge("east", "north", "top")).controlpoint,
                next(self.basis.edge("west", "north", "top")).controlpoint,
            ]
        )

    def clone(self) -> LRSplineVolume:
        return LRSplineVolume(self.w.copy())

    def insert_knot(self, new_knots: ScalarOrScalars, direction: Direction, multiplicity: int = 1) -> None:
        new_knots = _ensure_scalars(new_knots)
        direction = _check_direction(direction, self.pardim)
        for k in new_knots:
            start = list(self.start())
            end = list(self.end())
            start[direction] = end[direction] = k
            mr = raw.MeshRectangle(start[0], start[1], start[2], end[0], end[1], end[2], multiplicity)
            self.w.insert_line(mr)
        self.w.generateIDs()

    def insert(self, mr: MeshRect) -> None:
        self.w.insert_line(mr.w)

    @overload
    def evaluate(self, u: FArray, v: FArray, w: FArray, iel: int = ...) -> FArray: ...

    @overload
    def evaluate(self, u: float, v: float, w: float, iel: int = ...) -> FArray: ...

    def evaluate(self, u, v, w, iel=-1):  # type: ignore[no-untyped-def]
        if isinstance(u, np.ndarray) and isinstance(v, np.ndarray) and isinstance(w, np.ndarray):
            retval = np.array(
                [self.w.point(up, vp, wp, iEl=iel) for up, vp, wp in zip(u.flat, v.flat, w.flat)]
            )
            return retval.reshape(u.shape + (-1,))
        return self.w.point(u, v, w, iEl=iel)

    @overload
    def derivative(
        self, u: FArray, v: FArray, w: FArray, d: tuple[int, int, int] = ..., iel: int = ...
    ) -> FArray: ...

    @overload
    def derivative(
        self, u: float, v: float, w: float, d: tuple[int, int, int] = ..., iel: int = ...
    ) -> FArray: ...

    def derivative(self, u, v, w, d=(1, 1, 1), iel=-1):  # type: ignore[no-untyped-def]
        def wrapper(u: np.floating, v: np.floating, w: np.floating, nderivs: int) -> FArray:
            return self.w.point(u, v, w, nderivs, iEl=iel)

        return _derivative_helper((u, v, w), d, wrapper)

    def element_at(self, u: float, v: float, w: float) -> Element:  # type: ignore[override]
        return self.elements[self.w.getElementContaining(u, v, w)]

    __call__ = evaluate  # type: ignore[assignment]
