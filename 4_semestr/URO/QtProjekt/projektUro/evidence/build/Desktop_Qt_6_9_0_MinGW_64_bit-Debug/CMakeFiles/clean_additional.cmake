# Additional clean files
cmake_minimum_required(VERSION 3.16)

if("${CONFIG}" STREQUAL "" OR "${CONFIG}" STREQUAL "Debug")
  file(REMOVE_RECURSE
  "CMakeFiles\\evidence_autogen.dir\\AutogenUsed.txt"
  "CMakeFiles\\evidence_autogen.dir\\ParseCache.txt"
  "evidence_autogen"
  )
endif()
