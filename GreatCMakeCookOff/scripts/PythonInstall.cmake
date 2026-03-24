# Finds default PYTHON_PKG_DIR, as given by distutils (modification 3.13 not valid)
# Creates function to install relative to PYTHON_PKG_DIR
find_package(PythonInterp REQUIRED)

# Find python package directory
if(NOT DEFINED DEFAULT_PYTHON_PKG_DIR)
    execute_process(
      COMMAND ${PYTHON_EXECUTABLE} -c
          "import sysconfig; print(sysconfig.get_paths()['purelib'])"
          OUTPUT_VARIABLE DEFAULT_PYTHON_PKG_DIR
    )
    if(DEFAULT_PYTHON_PKG_DIR )
        string (STRIP ${DEFAULT_PYTHON_PKG_DIR} DEFAULT_PYTHON_PKG_DIR)
        set(DEFAULT_PYTHON_PKG_DIR
            ${DEFAULT_PYTHON_PKG_DIR} CACHE PATH "Main python package repository.")
        mark_as_advanced(DEFAULT_PYTHON_PKG_DIR)
    endif(DEFAULT_PYTHON_PKG_DIR)
endif()

if(NOT DEFINED PYTHON_PKG_DIR AND CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(PYTHON_PKG_DIR "${DEFAULT_PYTHON_PKG_DIR}")
elseif(NOT DEFINED PYTHON_PKG_DIR)
    set(PYTHON_PKG_DIR "${CMAKE_INSTALL_PREFIX}/lib/python")
    set(PYTHON_PKG_DIR "${PYTHON_PKG_DIR}${PYTHON_VERSION_MAJOR}.")
    set(PYTHON_PKG_DIR "${PYTHON_PKG_DIR}${PYTHON_VERSION_MINOR}/site-packages")
endif()
if(NOT DEFINED _OLD_PYTHON_PKG_DIR
    OR NOT "${PYTHON_PKG_DIR}" STREQUAL "${_OLD_PYTHON_PKG_DIR}")
    message(STATUS "Python install path (PYTHON_PKG_DIR): ${PYTHON_PKG_DIR}")
    set(_OLD_PYTHON_PKG_DIR "${PYTHON_PKG_DIR}" CACHE INTERNAL
        "Current python install path"
    )
endif()

# Installs relative to PYTHON_PKG_DIR
function(install_python)
    # Modify DESTINATION argument so that it points to the python directory,
    # unless path is absolute
    list(FIND ARGN DESTINATION destloc)
    if(NOT destloc EQUAL -1)
        math(EXPR destloc "${destloc} + 1")
        list(GET ARGN ${destloc} destination)
        if(NOT IS_ABSOLUTE ${destination})
            list(REMOVE_AT ARGN ${destloc})
            list(LENGTH ARGN length)
            if(${length} EQUAL ${destloc})
                list(APPEND ARGN "${PYTHON_PKG_DIR}/${destination}")
            else()
                list(INSERT ARGN ${destloc} "${PYTHON_PKG_DIR}/${destination}")
            endif()
        endif()
    endif()
    # Finally, calls normal install routine with modified argument list
    install(${ARGN})
endfunction()
