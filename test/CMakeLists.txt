include_directories(${PROJECT_SOURCE_DIR}/src)
add_executable(tinynn_test main.cpp)
target_link_libraries(tinynn_test tinynn)
install(TARGETS tinynn_test DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
