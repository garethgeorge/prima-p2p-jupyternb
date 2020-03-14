##############################################################################
# .. code-block:: python
#
#     class CircleElectrodeArray(ElectrodeArray):
#         """Electrodes arranged in a circle"""
#         ...
#
# This way, the ``CircleElectrodeArray`` class can access all public methods
# of :py:class:`~pulse2percept.implants.ElectrodeArray`.
#
# The constructor then has the job of creating all electrodes in the array
# and placing them at the appropriate location; for example, by using the
# :py:func:`~pulse2percept.implants.ElectrodeArray.add_electrodes` method.
#
# The constructor of the class should accept a number of arguments:
#
# - ``n_electrodes``: how many electrodes to arrange in a circle
# - ``radius``: the radius of the circle
# - ``x_center``: the x-coordinate of the center of the circle
# - ``y_center``: the y-coordinate of the center of the circle
#
# For simplicity, we will use :py:class:`~pulse2percept.implants.DiskElectrode`
# objects of a given radius (100um), although it would be relatively straightforward
# to allow the user to choose the electrode type.

import pulse2percept
from pulse2percept.viz import plot_implant_on_axon_map
import matplotlib.pyplot as plt
from pulse2percept.implants import ProsthesisSystem
from pulse2percept.implants import ElectrodeArray, DiskElectrode, PointSource, Electrode, ElectrodeGrid
from collections import OrderedDict, Sequence
import numpy as np
import math


class ElectrodeGridHex(ElectrodeArray):
    """Hexagonal grid of electrodes
    Parameters
    ----------
    shape : (rows, cols)
        A tuple containing the number of rows x columns in the grid
    spacing : double
        Electrode-to-electrode spacing in microns.
    x, y, z : double, optional, default: (0,0,0)
        3D coordinates of the center of the grid
    rot : double, optional, default: 0rad
        Rotation of the grid in radians (positive angle: counter-clockwise
        rotation on the retinal surface)
    names: (name_rows, name_cols), each of which either 'A' or '1'
        Naming convention for rows and columns, respectively.
        If 'A', rows or columns will be labeled alphabetically.
        If '1', rows or columns will be labeled numerically.
        Columns and rows may only be strings and integers.
        For example ('1', 'A') will number rows numerically and columns
        alphabetically.
    etype : :py:class:`~pulse2percept.implants.Electrode`, optional
        A valid Electrode class. By default,
        :py:class:`~pulse2percept.implants.PointSource` is used.
    **kwargs :
        Any additional arguments that should be passed to the
        :py:class:`~pulse2percept.implants.Electrode` constructor, such as
        radius ``r`` for :py:class:`~pulse2percept.implants.DiskElectrode`.
        See examples below.
    Examples
    --------
    An electrode grid with 2 rows and 4 columns, made of disk electrodes with
    10um radius spaced 20um apart, centered at (10, 20)um, and located 500um
    away from the retinal surface, with names like this:
        A1 A2 A3 A4
        B1 B2 B3 B4
    >>> from pulse2percept.implants import ElectrodeGrid, DiskElectrode
    >>> ElectrodeGrid((2, 4), 20, x=10, y=20, z=500, names=('A', '1'), r=10,
    ...               etype=DiskElectrode) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    ElectrodeGrid(..., name_cols='1',
                  name_rows='A', r=10..., rot=0..., shape=(2, 4),
                  spacing=20..., x=10..., y=20..., z=500...)
    There are three ways to access (e.g.) the last electrode in the grid,
    either by name (``grid['C3']``), by row/column index (``grid[2, 2]``), or
    by index into the flattened array (``grid[8]``):
    >>> from pulse2percept.implants import ElectrodeGrid
    >>> grid = ElectrodeGrid((3, 3), 20, names=('A', '1'))
    >>> grid['C3']  # doctest: +ELLIPSIS
    PointSource(x=20..., y=20..., z=0...)
    >>> grid['C3'] == grid[8] == grid[2, 2]
    True
    You can also access multiple electrodes at the same time by passing a
    list of indices/names (it's ok to mix-and-match):
    >>> from pulse2percept.implants import ElectrodeGrid, DiskElectrode
    >>> grid = ElectrodeGrid((3, 3), 20, etype=DiskElectrode, r=10)
    >>> grid[['A1', 1, (0, 2)]]  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [DiskElectrode(r=10..., x=-20.0, y=-20.0, z=0...),
     DiskElectrode(r=10..., x=0.0, y=-20.0, z=0...),
     DiskElectrode(r=10..., x=20.0, y=-20.0, z=0...)]
    """

    def __init__(self, shape, spacing, x=0, y=0, z=0, rot=0, names=('A', '1'),
                 etype=PointSource, **kwargs):
        if not isinstance(names, (tuple, list, np.ndarray)):
            raise TypeError("'names' must be a tuple/list of (rows, cols)")
        if not isinstance(shape, (tuple, list, np.ndarray)):
            raise TypeError("'shape' must be a tuple/list of (rows, cols)")
        if len(shape) != 2:
            raise ValueError("'shape' must have two elements: (rows, cols)")
        if np.prod(shape) <= 0:
            raise ValueError("Grid must have all non-zero rows and columns.")
        if not issubclass(etype, Electrode):
            raise TypeError("'etype' must be a valid Electrode object.")
        if issubclass(etype, DiskElectrode):
            if 'r' not in kwargs.keys():
                raise ValueError("A DiskElectrode needs a radius ``r``.")
            self.r = kwargs['r']

        # Extract rows and columns from shape:
        self.shape = shape
        self.x = x
        self.y = y
        self.z = z
        self.rot = rot
        self.spacing = spacing
        self.etype = etype  # add etype variable under the class
        # Store names for rows/cols separately:
        if len(names) == 2:
            self.name_rows, self.name_cols = names
        self.names = names
        # Instantiate empty collection of electrodes. This dictionary will be
        # populated in a private method ``_set_egrid``:
        self.electrodes = OrderedDict()
        self._set_grid()

    def get_params(self):
        """Return a dictionary of class attributes"""
        params = {'shape': self.shape, 'spacing': self.spacing,
                  'x': self.x, 'y': self.y, 'z': self.z,
                  'rot': self.rot, 'etype': self.etype,
                  'name_cols': self.name_cols, 'name_rows': self.name_rows}
        if issubclass(self.etype, DiskElectrode):
            params.update({'r': self.r})
        return params

    def __getitem__(self, item):
        """Access electrode(s) in the grid
        Parameters
        ----------
        item : index, string, tuple, or list thereof
            An electrode in the grid can be accessed in three ways:
            *  by name, e.g. grid['A1']
            *  by index into the flattened array, e.g. grid[0]
            *  by (row, column) index into the 2D grid, e.g. grid[0, 0]
            You can also pass a list or NumPy array of the above.
        Returns
        -------
        electrode : `~pulse2percept.implants.Electrode`, list thereof, or None
            Returns the corresponding `~pulse2percept.implants.Electrode`
            object or ``None`` if index is not valid.
        """
        if isinstance(item, (list, np.ndarray)):
            # Recursive call for list items:
            return [self.__getitem__(i) for i in item]
        try:
            # Access by key into OrderedDict, e.g. grid['A1']:
            return self.electrodes[item]
        except (KeyError, TypeError):
            # Access by index into flattened array, e.g. grid[0]:
            try:
                return list(self.electrodes.values())[item]
            except (KeyError, TypeError):
                # Access by [r, c] into 2D grid, e.g. grid[0, 3]:
                try:
                    idx = np.ravel_multi_index(item, self.shape)
                    return list(self.electrodes.values())[idx]
                except (KeyError, ValueError):
                    # Index not found:
                    return None

    def _set_grid(self):
        """Private method to build the electrode grid"""

        n_elecs = np.prod(self.shape)
        rows, cols = self.shape

        # The user did not specify a unique naming scheme:
        if len(self.names) == 2:
            # Create electrode names, using either A-Z or 1-n:
            if self.name_rows.isalpha():
                rws = [chr(i) for i in range(
                    ord(self.name_rows), ord(self.name_rows) + rows + 1)]
            elif self.name_rows.isdigit():
                rws = [str(i) for i in range(
                    int(self.name_rows), rows + int(self.name_rows))]
            else:
                raise ValueError("rows must be alphabetic or numeric")

            if self.name_cols.isalpha():
                clms = [chr(i) for i in range(ord(self.name_cols),
                                              ord(self.name_cols) + cols)]
            elif self.name_cols.isdigit():
                clms = [str(i) for i in range(
                    int(self.name_cols), cols + int(self.name_cols))]
            else:
                raise ValueError("Columns must be alphabetic or numeric.")

            # facilitating Argus I naming scheme
            if self.name_cols.isalpha() and not self.name_rows.isalpha():
                names = [clms[j] + rws[i] for i in range(len(rws))
                         for j in range(len(clms))]
            else:
                names = [rws[i] + clms[j] for i in range(len(rws))
                         for j in range(len(clms))]
        else:
            if len(self.names) != n_elecs:
                raise ValueError("If `names` specifies more than row/column "
                                 "names, it must have %d entries, not "
                                 "%d)." % (n_elecs, len(self.names)))
            names = self.names

        if isinstance(self.z, (list, np.ndarray)):
            # Specify different height for every electrode in a list:
            z_arr = np.asarray(self.z).flatten()
            if z_arr.size != n_elecs:
                raise ValueError("If `h` is a list, it must have %d entries, "
                                 "not %d." % (n_elecs, len(self.z)))
        else:
            # If `z` is a scalar, choose same height for all electrodes:
            z_arr = np.ones(n_elecs, dtype=float) * self.z

        # Make a 2D meshgrid from x, y coordinates:
        # For example, cols=3 with spacing=100 should give: [-100, 0, 100]
        x_arr_lshift = (np.arange(cols) * self.spacing -
                        (cols / 2.0 - 0.5) * self.spacing - self.spacing * 0.25)
        x_arr_rshift = (np.arange(cols) * self.spacing -
                        (cols / 2.0 - 0.5) * self.spacing + self.spacing * 0.25)
        y_arr = (np.arange(rows) * math.sqrt(3) * self.spacing/2.0 -
                 (rows / 2.0 - 0.5) * self.spacing)
        x_arr_lshift, y_arr_lshift = np.meshgrid(
            x_arr_lshift, y_arr, sparse=False)
        x_arr_rshift, y_arr_rshift = np.meshgrid(
            x_arr_rshift, y_arr, sparse=False)

        # added code to interleave arrays
        x_arr = []
        for row in range(0, rows):
            if row % 2 == 0:
                x_arr.append(x_arr_lshift[row])
            else:
                x_arr.append(x_arr_rshift[row])
        x_arr = np.array(x_arr)
        y_arr = y_arr_rshift

        # Rotate the grid:
        rotmat = np.array([np.cos(self.rot), -np.sin(self.rot),
                           np.sin(self.rot), np.cos(self.rot)]).reshape((2, 2))
        xy = np.matmul(rotmat, np.vstack((x_arr.flatten(), y_arr.flatten())))
        x_arr = xy[0, :]
        y_arr = xy[1, :]

        # Apply offset to make the grid centered at (self.x, self.y):
        x_arr += self.x
        y_arr += self.y

        if issubclass(self.etype, DiskElectrode):
            if isinstance(self.r, (list, np.ndarray)):
                # Specify different radius for every electrode in a list:
                if len(self.r) != n_elecs:
                    err_s = ("If `r` is a list, it must have %d entries, not "
                             "%d)." % (n_elecs, len(self.r)))
                    raise ValueError(err_s)
                r_arr = self.r
            else:
                # If `r` is a scalar, choose same radius for all electrodes:
                r_arr = np.ones(n_elecs, dtype=float) * self.r

            # Create a grid of DiskElectrode objects:
            for x, y, z, r, name in zip(x_arr, y_arr, z_arr, r_arr, names):
                self.add_electrode(name, DiskElectrode(x, y, z, r))
        elif issubclass(self.etype, PointSource):
            # Create a grid of PointSource objects:
            for x, y, z, name in zip(x_arr, y_arr, z_arr, names):
                self.add_electrode(name, PointSource(x, y, z))
        else:
            raise NotImplementedError


class Prima(ProsthesisSystem):
    def __init__(self, x=0, y=0, z=0, rot=0, eye='RE', stim=None,
                 use_legacy_names=False):
        # Prima is a 378 electrode implant measuring 2mm by 2mm
        # https://www.edisongroup.com/wp-content/uploads/2019/04/Pixium-Vision-Prima-gearing-up-for-a-pivotal-year.pdf
        # in its current iteration, we approximate this as ~400 electrodes
        self.eye = eye
        self.shape = (20, 20)

        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3419261/ <- describes active vs return electrodes
        # so the hexagonal electrode consides of an active electrode surrounded by a ring return electrode 
        # this makes it a very different type of electrode from the basic disk electrode...
        # for now, based on data available, we estimate that the density is ~100um spacing with an active electrode of 70um
        r_arr = 10
        spacing = 105

        self.earray = ElectrodeGridHex(self.shape, spacing, x=x, y=y, z=z,
                                       rot=rot, etype=DiskElectrode, r=r_arr)

        # Set stimulus if available:
        self.stim = stim

        # Set left/right eye:
        if not isinstance(eye, str):
            raise TypeError("'eye' must be a string, either 'LE' or 'RE'.")
        if eye != 'LE' and eye != 'RE':
            raise ValueError("'eye' must be either 'LE' or 'RE'.")
        
        self.eye = eye
        # Unfortunately, in the left eye the labeling of columns is reversed...
        if eye == 'LE':
            # FIXME: Would be better to have more flexibility in the naming
            # convention. This is a quick-and-dirty fix:
            names = list(self.earray.keys())
            objects = list(self.earray.values())
            names = np.array(names).reshape(self.earray.shape)
            # Reverse column names:
            for row in range(self.earray.shape[0]):
                names[row] = names[row][::-1]
            # Build a new ordered dict:
            electrodes = OrderedDict([])
            for name, obj in zip(names.ravel(), objects):
                electrodes.update({name: obj})
            # Assign the new ordered dict to earray:
            self.earray.electrodes = electrodes

    def get_params(self):
        params = super().get_params()
        params.update({'shape': self.shape})
        return params


# class HexElectrode(DiskElectrode):
#     def electric_potential(self, x, y, z, v0):
#         """Calculate electric potential at (x, y, z)

#         Parameters
#         ----------
#         x/y/z : double
#             3D location at which to evaluate the electric potential
#         v0 : double
#             The quasi-static disk potential relative to a ground electrode at
#             infinity

#         Returns
#         -------
#         pot : double
#             The electric potential at (x, y, z).


#         The electric potential :math:`V(r,z)` of a disk electrode is given by
#         [WileyWebster1982]_:

#         .. math::

#             V(r,z) = \\sin^{-1} \\bigg\\{ \\frac{2a}{\\sqrt{(r-a)^2 + z^2} + \\sqrt{(r+a)^2 + z^2}} \\bigg\\} \\times \\frac{2 V_0}{\\pi},

#         for :math:`z \\neq 0`, where :math:`r` and :math:`z` are the radial
#         and axial distances from the center of the disk, :math:`V_0` is the
#         disk potential, :math:`\\sigma` is the medium conductivity,
#         and :math:`a` is the disk radius.

#         """
#         radial_dist = np.sqrt((x - self.x) ** 2 + (y - self.y) ** 2)
#         axial_dist = z - self.z
#         if np.isclose(axial_dist, 0):
#             # Potential on the electrode surface (Eq. 9 in Wiley & Webster):
#             if radial_dist > self.r:
#                 # Outside the electrode:
#                 return 2.0 * v0 / np.pi * np.arcsin(self.r / radial_dist)
#             else:
#                 # On the electrode:
#                 return v0
#         else:
#             # Off the electrode surface (Eq. 10):
#             numer = 2 * self.r
#             denom = np.sqrt((radial_dist - self.r) ** 2 + axial_dist ** 2)
#             denom += np.sqrt((radial_dist + self.r) ** 2 + axial_dist ** 2)
#             return 2.0 * v0 / np.pi * np.arcsin(numer / denom)
