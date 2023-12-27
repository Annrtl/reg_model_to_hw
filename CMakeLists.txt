cmake_minimum_required(VERSION 3.20)
project( nn )

find_package(h5cpp REQUIRED)
include_directories( include )
include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

add_executable ( nn src/main.cpp )

target_sources( nn PRIVATE src/dense.cpp )
target_sources( nn PRIVATE src/import.cpp )

target_link_libraries( nn jsoncpp )
target_link_libraries( nn h5cpp::h5cpp )

#Convert files to header
set(H5_BINARY_FILE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/model/model.h5)
set(H5_HEADER_FILE_PATH ${CMAKE_CURRENT_BINARY_DIR}/model.h5.h)
set(JSON_BINARY_FILE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/model/model.json)
set(JSON_HEADER_FILE_PATH ${CMAKE_CURRENT_BINARY_DIR}/model.json.h)

add_custom_command(
    OUTPUT ${H5_HEADER_FILE_PATH}
    COMMAND xxd -i -n "h5_model_bin" ${H5_BINARY_FILE_PATH} > ${H5_HEADER_FILE_PATH}
    DEPENDS ${H5_BINARY_FILE_PATH}
    COMMENT "Converting binary file to C++ header"
)

add_custom_command(
    OUTPUT ${JSON_HEADER_FILE_PATH}
    COMMAND xxd -i -n "json_model_bin" ${JSON_BINARY_FILE_PATH} > ${JSON_HEADER_FILE_PATH}
    DEPENDS ${JSON_BINARY_FILE_PATH}
    COMMENT "Converting binary file to C++ header"
)

add_custom_target(ConvertH5Model
    ALL
    DEPENDS ${H5_HEADER_FILE_PATH}
)

add_custom_target(convertJsonModel
    ALL
    DEPENDS ${JSON_HEADER_FILE_PATH}
)

add_dependencies( nn convertJsonModel ConvertH5Model )