cmake_minimum_required(VERSION 3.16)
project(SeafEnergyKernel LANGUAGES CXX)

add_library(sheaf_energy INTERFACE)
target_include_directories(sheaf_energy INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(sheaf_energy INTERFACE cxx_std_17)

enable_testing()

add_executable(sheaf_energy_test tests/cpp/sheaf_energy_test.cpp)
target_link_libraries(sheaf_energy_test PRIVATE sheaf_energy)
add_test(NAME sheaf_energy_test COMMAND sheaf_energy_test)

if(MSVC)
    target_compile_options(sheaf_energy_test PRIVATE /W4)
else()
    target_compile_options(sheaf_energy_test PRIVATE -Wall -Wextra -Wpedantic)
endif()
