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

# Заголовки подключаются как <onnxruntime/...>, а пакет объявляет сам каталог
# onnxruntime — поэтому добавляем тот, что на уровень выше.
get_target_property(ONNXRUNTIME_INCLUDE_DIR onnxruntime::onnxruntime INTERFACE_INCLUDE_DIRECTORIES)
get_filename_component(ONNXRUNTIME_INCLUDE_ROOT ${ONNXRUNTIME_INCLUDE_DIR} DIRECTORY)

target_include_directories(smileDetector PRIVATE
    ${ONNXRUNTIME_INCLUDE_ROOT})

target_link_libraries(smileDetector 
    ${OpenCV_LIBS}
    onnxruntime::onnxruntime )
