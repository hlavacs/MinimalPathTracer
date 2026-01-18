cmake_minimum_required(VERSION 3.10)

project(minimal_pt LANGUAGES CXX)

# Allow dependencies with older cmake_minimum_required to configure under newer CMake.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "Minimum CMake policy version for dependencies" FORCE)

include(FetchContent)
FetchContent_Declare(
  glm
  GIT_REPOSITORY https://github.com/g-truc/glm.git
  GIT_TAG 0.9.9.8
)
FetchContent_MakeAvailable(glm)

add_executable(minimal_pt minimal_pt.cpp)
target_link_libraries(minimal_pt PRIVATE glm::glm)

set_target_properties(minimal_pt PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)

if (MSVC)
  target_compile_options(minimal_pt PRIVATE
    $<$<CONFIG:Release>:/O2 /openmp>
    $<$<CONFIG:RelWithDebInfo>:/O2 /openmp>
    $<$<CONFIG:MinSizeRel>:/O2 /openmp>
  )
else()
  target_compile_options(minimal_pt PRIVATE -O3)
endif()

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
  target_link_libraries(minimal_pt PRIVATE OpenMP::OpenMP_CXX)
endif()
