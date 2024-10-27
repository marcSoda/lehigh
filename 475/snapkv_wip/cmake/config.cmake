function(CHECK_CUDA_VERSION CUDA_MIN_VERSION CUDA_MAX_VERSION)
    if(NOT CUDA_VERSION)
        set(CUDA_VERSION ${CMAKE_CUDA_COMPILER_VERSION} PARENT_SCOPE)
    endif()
    if(NOT CUDA_TOOLKIT_ROOT_DIR)
        get_filename_component(CUDA_TOOLKIT_ROOT_DIR "${CMAKE_CUDA_COMPILER}/../.." ABSOLUTE)
    endif()
    
    if(CUDA_VERSION VERSION_LESS CUDA_MIN_VERSION AND CUDA_VERSION VERSION_GREATER CUDA_MAX_VERSION)
        message(FATAL_ERROR "CUDA ${CUDA_MIN_VERSION}-${CUDA_MAX_VERSION} required, found ${CUDA_VERSION}")
    endif()
endfunction()

function(SET_CONSISTENT_STANDARD STANDARD)
    set(CMAKE_CXX_STANDARD ${STANDARD})
    set(CMAKE_CUDA_STANDARD ${STANDARD})
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endfunction()

macro(SETUP_CONAN)
    if(NOT USING_CONAN)
        message(STATUS "Running conan install ${CMAKE_SOURCE_DIR} -if ${CMAKE_BINARY_DIR} --build=missing")
        execute_process(
            COMMAND conan install -s compiler.libcxx=libstdc++17 -s build_type=${CMAKE_BUILD_TYPE} ${CMAKE_SOURCE_DIR} -if ${CMAKE_BINARY_DIR} --build=missing
            RESULT_VARIABLE conan_code)
        if(NOT "${conan_code}" STREQUAL "0")
            message(FATAL_ERROR "Conan failed ${conan_code}")
        endif()
    endif()
    include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    conan_basic_setup(TARGETS)
endmacro()

function(SET_SPDLOG LIBRARY_OR_EXE)
    target_compile_options(${LIBRARY_OR_EXE} PUBLIC "$<$<CONFIG:DEBUG>:-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE>")
    target_compile_options(${LIBRARY_OR_EXE} PUBLIC "$<$<CONFIG:RELEASE>:-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_ERROR>")
    target_compile_options(${LIBRARY_OR_EXE} PUBLIC "$<$<CONFIG:RELWITHDEBINFO>:-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_ERROR>")
endfunction()

function(add_test_executable TARGET SOURCE)

    set(options DISABLE_TEST)
    set(oneValueArgs "")
    set(multiValueArgs LIBS)

    cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    add_executable(${TARGET} ${SOURCE})
    add_executable("asan${TARGET}" ${SOURCE})
    add_executable("cov${TARGET}" ${SOURCE})
    add_executable("dbg${TARGET}" ${SOURCE})

    target_link_libraries("asan${TARGET}" PRIVATE asan)
    target_link_libraries("cov${TARGET}" PRIVATE gcov)

    foreach(NAME "${TARGET}" "asan${TARGET}" "cov${TARGET}" "dbg${TARGET}")
        target_link_libraries(${NAME} PRIVATE sikv ${__LIBS})
        target_compile_options(${NAME} PRIVATE -DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE)
    endforeach()

    foreach(NAME "asan${TARGET}" "cov${TARGET}" "dbg${TARGET}")
        target_compile_options(${NAME} PRIVATE -DCULOG_ENABLE)
    endforeach()

    target_compile_options("cov${TARGET}" PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler '--coverage'>)
    target_compile_options("cov${TARGET}" PRIVATE $<$<COMPILE_LANGUAGE:CXX>:--coverage>)
    target_compile_options("asan${TARGET}" PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -fsanitize=address>)
    target_compile_options("asan${TARGET}" PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler "-fsanitize=address">)
    target_compile_options("dbg${TARGET}" PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -G -g>)
    if(NOT ${__DISABLE_TEST})
        add_test(NAME ${TARGET} COMMAND ${TARGET} --catch_system_errors=false)
        set_property(TEST ${TARGET} APPEND PROPERTY ENVIRONMENT "BUILD_ROOT=${CMAKE_BINARY_DIR}/")
    endif()
endfunction()

macro(doxygen)
    
    find_package(Doxygen)

    if(DOXYGEN_FOUND)
        file(GLOB_RECURSE dependsInclude ${CMAKE_SOURCE_DIR}/include/*)
        file(GLOB_RECURSE dependsExperiment ${CMAKE_SOURCE_DIR}/experiment/*)
        file(GLOB_RECURSE dependsTPCC ${CMAKE_BINARY_DIR}/tpccInclude/*)
        file(GLOB_RECURSE dependsSystem ${CMAKE_SOURCE_DIR}/system/*)

        configure_file(${CMAKE_SOURCE_DIR}/Doxygen ${CMAKE_BINARY_DIR}/Doxygen COPYONLY)
        add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/html/index.html
                           DEPENDS ${CMAKE_BINARY_DIR}/Doxygen ${dependsInclude} ${dependsSystem} ${dependsTPCC} ${dependsExperiment} ${CMAKE_SOURCE_DIR}/README.md
                           COMMAND doxygen ${CMAKE_BINARY_DIR}/Doxygen
                           COMMENT "Generating docs")
        add_custom_target(DoxygenSIKV ALL DEPENDS ${CMAKE_BINARY_DIR}/html/index.html)
    endif()
endmacro()
