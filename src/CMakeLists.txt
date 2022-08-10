aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRC)
add_library(tinynn SHARED ${SRC})
install(TARGETS tinynn DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(FILES allocator.hpp tensor.hpp DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
