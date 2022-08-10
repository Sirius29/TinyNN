cmake_minimum_required(VERSION 3.0.0)
project(tinynn VERSION 0.1.0)

include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR})
add_subdirectory(src)
add_subdirectory(example)
add_subdirectory(test)
