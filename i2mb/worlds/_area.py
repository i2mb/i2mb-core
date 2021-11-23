#  I2MB
#  Copyright (C) 2021  FAU - RKI
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
import numpy as np


class ExitInit(type):
    def __call__(cls, *args, **kwargs):
        cls_instance = cls.__new__(cls)
        cls_instance.__init__(*args, **kwargs)
        cls_instance.__exit_init_method__()
        return cls_instance

    def exit_method(self):
        pass


class Area(metaclass=ExitInit):
    __num_instances = 0
    __id_map = {}

    def __init__(self, dims=None, height=None, width=None, origin=None, rotation=0, scale=1, subareas=None):
        if subareas is None:
            subareas = []

        self.__sub_areas = []
        self.__sub_areas.append(subareas)
        self.__dims = np.array([0., 0.])
        self.__origin = np.array([0., 0.])
        self.__opposite = np.array([0., 0.])
        if width is None:
            width = 1.

        if height is None:
            height = 1.

        if dims is None:
            self.__dims[:] = width, height
        else:
            self.__dims[:] = dims

        width, height = self.dims
        self.__bounding_box = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=float)

        self.scale = scale
        self.dims *= scale

        self.__points = []
        self.__points.extend(self.__bounding_box)

        self.__update_dims()

        self.rotation = rotation
        self.origin = origin

        self.parent = None
        self.id = Area.__num_instances
        Area.__num_instances += 1
        Area.__id_map[self.id] = self

    @property
    def opposite(self):
        return self.__opposite

    @property
    def width(self):
        return self.__dims[0]

    @width.setter
    def width(self, value):
        self.dims = value, self.height

    @property
    def height(self):
        return self.__dims[1]

    @height.setter
    def height(self, value):
        self.dims = self.width, value

    @property
    def dims(self):
        return self.__dims

    @dims.setter
    def dims(self, value):
        if value is None:
            self.__dims = np.array([1., 1.])
            return

        self.__dims = np.array(value)
        self.__bounding_box[[1, 2], 0] = self.__dims[0]
        self.__bounding_box[[2, 3], 1] = self.__dims[1]
        self.__bounding_box[[0, 3], 0] = 0.
        self.__bounding_box[[0, 1], 1] = 0.

    @property
    def points(self):
        return self.__points

    @points.setter
    def points(self, value):
        assert len(value) == len(self.points), "Trying to set a point vector with the wrong number of points"
        for r_ix, point in enumerate(value):
            # set the value without modifying list pointers
            for c_ix, v in enumerate(point):
                self.points[r_ix][c_ix] = v

    @property
    def origin(self):
        return self.__origin

    @origin.setter
    def origin(self, origin):
        if origin is None:
            new_origin = np.array([0., 0.])
        else:
            new_origin = np.array(origin)

        self.__origin[:] = new_origin.ravel()
        self.__opposite[:] = self.__origin + self.dims

    def __rotate(self, rotation):
        self.rotation = (rotation + self.rotation) % 360.
        rotation = np.deg2rad(rotation)
        R = np.array([[np.cos(rotation), -np.sin(rotation)],
                      [np.sin(rotation), np.cos(rotation)]])
        o = np.atleast_2d([self.width / 2., self.height / 2.])
        # o = np.atleast_2d([0, 0])

        points = np.atleast_2d(self.points)
        self.points = ((R @ (points.T - o.T) + o.T).T)
        self.__update_internal_origin()
        self.__update_dims()

    def __update_dims(self):
        new_dims = self.__bounding_box.max(axis=0) - self.__bounding_box.min(axis=0)
        self.dims[:] = new_dims

    def __update_internal_origin(self):
        new_origin = self.__bounding_box.min(axis=0)
        if (new_origin == np.array([0, 0])).all():
            return

        self.points = self.points - new_origin

    def update_external_origin(self):
        p1, p2 = self.opposite, self.origin
        p3 = [p1[0], p2[1]]
        p4 = [p2[0], p1[1]]

        bbox = np.array([p1, p2, p3, p4])
        new_origin = bbox.min(axis=0)
        self.origin = new_origin
        self.opposite[:] = self.origin + self.dims
        return

    def rotate(self, rotation):
        self.__rotate(rotation)
        self.update_external_origin()
        for sub_area in self.__sub_areas:
            # region.rotation = 0
            sub_area.rotate(rotation)
            sub_area.update_external_origin()

    def register_sub_areas(self, subareas):
        # This method only makes sense to run in init clauses.
        self.__sub_areas.append(subareas)

    def __exit_init_method__(self):
        # Flatten subareas
        self.__sub_areas = [s for s_areas in self.__sub_areas for s in s_areas]

        # Make sure every thing is properly oriented
        rotation = self.rotation
        self.rotate(self.rotation)

        # rotate() is cumulative, so we have to correct for the first rotation.
        self.rotation = rotation

        # Update external origin for rotated geometry
        self.update_external_origin()

    def list_all_regions(self):
        area_list = [self]
        for r in self.__sub_areas:
            area_list.extend(r.list_all_regions())

        return area_list


