//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <DetourNavMesh.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/eigen/dense.h>

namespace nb = nanobind;
using namespace nb::literals;

void bindDetourNavMesh(nb::module_& m) {
    nb::class_<dtNavMesh>(m, "dtNavMesh")
            .def(nb::init<>())
            .def("init", nb::overload_cast<const dtNavMeshParams*>(&dtNavMesh::init), "params"_a)
            .def("getParams", &dtNavMesh::getParams)
            .def("getTileAt", &dtNavMesh::getTileAt);
}