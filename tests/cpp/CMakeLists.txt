#  Copyright (c) 2023 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.

# create sample app project
project(cpp-tests LANGUAGES C CXX)

set(SRC_FILES
        test_variable.cpp
)

add_executable(${PROJECT_NAME} ${SRC_FILES})

find_package(GTest CONFIG REQUIRED)

target_include_directories(${PROJECT_NAME} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/libs
        ${CPPAD_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
        GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main
)