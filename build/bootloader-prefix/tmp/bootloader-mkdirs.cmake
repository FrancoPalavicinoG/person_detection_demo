# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/Users/franco_palavicino/esp/idf/esp-idf/components/bootloader/subproject"
  "/Users/franco_palavicino/esp/projects_tf/person_detection_proyect/build/bootloader"
  "/Users/franco_palavicino/esp/projects_tf/person_detection_proyect/build/bootloader-prefix"
  "/Users/franco_palavicino/esp/projects_tf/person_detection_proyect/build/bootloader-prefix/tmp"
  "/Users/franco_palavicino/esp/projects_tf/person_detection_proyect/build/bootloader-prefix/src/bootloader-stamp"
  "/Users/franco_palavicino/esp/projects_tf/person_detection_proyect/build/bootloader-prefix/src"
  "/Users/franco_palavicino/esp/projects_tf/person_detection_proyect/build/bootloader-prefix/src/bootloader-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/franco_palavicino/esp/projects_tf/person_detection_proyect/build/bootloader-prefix/src/bootloader-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/franco_palavicino/esp/projects_tf/person_detection_proyect/build/bootloader-prefix/src/bootloader-stamp${cfgdir}") # cfgdir has leading slash
endif()
