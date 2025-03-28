from numpydantic import NDArray, Shape

BASE_TYPES = (int, float, str)
IMAGE = NDArray[Shape["* x, * y"], float]
POINTS = NDArray[Shape["* x, 2 y"], float]
POINT = NDArray[Shape["2 x"], float]
R_THETA = NDArray[Shape["2 x"], float]
COORDINATES = NDArray[Shape["* x, 2 y"], int]
COORDINATE = NDArray[Shape["* x"], int]
X = NDArray[Shape["* x"], float]
Y = NDArray[Shape["* x"], float]
THETA = NDArray[Shape["* x"], float]
R = NDArray[Shape["* x"], float]
SIGNAL = NDArray[Shape["* x"], float]
POINTS_AND_SIGNAL = NDArray[Shape["* x, 3 y"], float]
BINS_LIMITS = NDArray[Shape["* x"], float]
