function (add_stub name)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "VERBOSE;INCLUDE_PRIVATE;EXCLUDE_DOCSTRINGS;INSTALL_TIME;EXCLUDE_FROM_ALL" "MODULE;OUTPUT;MARKER_FILE;COMPONENT;PATTERN_FILE" "PYTHON_PATH;DEPENDS")

    if (EXISTS ${NB_DIR}/src/stubgen.py)
        set(NB_STUBGEN "${NB_DIR}/src/stubgen.py")
    elseif (EXISTS ${NB_DIR}/stubgen.py)
        set(NB_STUBGEN "${NB_DIR}/stubgen.py")
    else()
        message(FATAL_ERROR "nanobind_add_stub(): could not locate 'stubgen.py'!")
    endif()

    if (NOT ARG_VERBOSE)
        list(APPEND NB_STUBGEN_ARGS -q)
    else()
        set(NB_STUBGEN_EXTRA USES_TERMINAL)
    endif()

    if (ARG_INCLUDE_PRIVATE)
        list(APPEND NB_STUBGEN_ARGS -P)
    endif()

    if (ARG_EXCLUDE_DOCSTRINGS)
        list(APPEND NB_STUBGEN_ARGS -D)
    endif()

    foreach (TMP IN LISTS ARG_PYTHON_PATH)
        list(APPEND NB_STUBGEN_ARGS -i "${TMP}")
    endforeach()

    if (ARG_PATTERN_FILE)
        list(APPEND NB_STUBGEN_ARGS -p "${ARG_PATTERN_FILE}")
    endif()

    if (ARG_MARKER_FILE)
        list(APPEND NB_STUBGEN_ARGS -M "${ARG_MARKER_FILE}")
        list(APPEND NB_STUBGEN_OUTPUTS "${ARG_MARKER_FILE}")
    endif()

    if (NOT ARG_MODULE)
        message(FATAL_ERROR "nanobind_add_stub(): a 'MODULE' argument must be specified!")
    else()
        list(APPEND NB_STUBGEN_ARGS -m "${ARG_MODULE}")
    endif()
    list(APPEND NB_STUBGEN_ARGS -r)

    if (NOT ARG_OUTPUT)
        message(FATAL_ERROR "nanobind_add_stub(): an 'OUTPUT' argument must be specified!")
    else()
        list(APPEND NB_STUBGEN_ARGS -O "${ARG_OUTPUT}")
        list(APPEND NB_STUBGEN_OUTPUTS "${ARG_OUTPUT}")
    endif()

    file(TO_CMAKE_PATH ${Python_EXECUTABLE} NB_Python_EXECUTABLE)

    set(NB_STUBGEN_CMD "${NB_Python_EXECUTABLE}" "${NB_STUBGEN}" ${NB_STUBGEN_ARGS})

    if (NOT ARG_INSTALL_TIME)
        add_custom_command(
                OUTPUT ${NB_STUBGEN_OUTPUTS}
                COMMAND ${NB_STUBGEN_CMD}
                WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                DEPENDS ${ARG_DEPENDS} "${NB_STUBGEN}" "${ARG_PATTERN_FILE}"
                ${NB_STUBGEN_EXTRA}
        )
        add_custom_target(${name} ALL DEPENDS ${NB_STUBGEN_OUTPUTS})
    else()
        set(NB_STUBGEN_EXTRA "")
        if (ARG_COMPONENT)
            list(APPEND NB_STUBGEN_EXTRA COMPONENT ${ARG_COMPONENT})
        endif()
        if (ARG_EXCLUDE_FROM_ALL)
            list(APPEND NB_STUBGEN_EXTRA EXCLUDE_FROM_ALL)
        endif()
        # \${CMAKE_INSTALL_PREFIX} has same effect as $<INSTALL_PREFIX>
        # This is for compatibility with CMake < 3.27.
        # For more info: https://github.com/wjakob/nanobind/issues/420#issuecomment-1971353531
        install(CODE "set(CMD \"${NB_STUBGEN_CMD}\")\nexecute_process(\n COMMAND \$\{CMD\}\n WORKING_DIRECTORY \"\${CMAKE_INSTALL_PREFIX}\"\n)" ${NB_STUBGEN_EXTRA})
    endif()
endfunction()