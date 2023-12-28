cmake_minimum_required(VERSION 3.20)
project( nn )

include_directories( include )
include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

add_executable ( nn src/main.cpp )

target_sources( nn PRIVATE src/dense.cpp )
target_sources( nn PRIVATE src/import.cpp )

target_link_libraries( nn jsoncpp )

#Convert files to header
set(JSON_BINARY_FILE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/model/model.json)
set(JSON_HEADER_FILE_PATH ${CMAKE_CURRENT_BINARY_DIR}/model.json.h)

add_custom_command(
    OUTPUT ${JSON_HEADER_FILE_PATH}
    COMMAND xxd -i -n "json_model_bin" ${JSON_BINARY_FILE_PATH} > ${JSON_HEADER_FILE_PATH}
    DEPENDS ${JSON_BINARY_FILE_PATH}
    COMMENT "Converting binary file to C++ header"
)

add_custom_target(convertJsonModel
    ALL
    DEPENDS ${JSON_HEADER_FILE_PATH}
)

add_dependencies( nn convertJsonModel )