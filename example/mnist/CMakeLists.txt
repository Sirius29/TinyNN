include_directories(${PROJECT_SOURCE_DIR}/src)
add_executable(mnist main.cpp)
target_link_libraries(mnist tinynn)
install(TARGETS mnist DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)