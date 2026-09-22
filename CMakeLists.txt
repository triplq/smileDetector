cmake_minimum_required(VERSION 3.10)

project( smileDetector )
set(CMAKE_CXX_STANDARD 17)

find_package( OpenCV REQUIRED )
find_package( onnxruntime REQUIRED )

include_directories( 
    ${OpenCV_INCLUDE_DIRS})

add_executable(smileDetector 
    main.cpp )

# Модель и картинки ищутся относительно рабочего каталога (./Resources),
# поэтому копируем их рядом с исполняемым файлом.
file(COPY ${CMAKE_SOURCE_DIR}/Resources DESTINATION ${CMAKE_BINARY_DIR})

set_target_properties(smileDetector PROPERTIES
    BUILD_RPATH "/usr/local/lib"
    INSTALL_RPATH "/usr/local/lib"
)

target_link_libraries(smileDetector 
    ${OpenCV_LIBS}
    onnxruntime )
