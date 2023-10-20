/*!
 * Copyright (c) 2023 by Contributors
 * \file file_utils.h
 * \brief Helper functions for manipulating files
 * \author Hyunsu Cho
 */

#ifndef TREELITE_DETAIL_FILE_UTILS_H_
#define TREELITE_DETAIL_FILE_UTILS_H_

#include <treelite/logging.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

namespace treelite::detail {

inline std::ifstream OpenFileForReadAsStream(std::filesystem::path const& filepath) {
  auto path = std::filesystem::weakly_canonical(filepath);
  TREELITE_CHECK(std::filesystem::exists(path)) << "Path " << filepath << " does not exist";
  std::ifstream ifs(path, std::ios::in | std::ios::binary);
  TREELITE_CHECK(ifs) << "Could not open file " << filepath;
  ifs.exceptions(std::ios::badbit);  // Throw exceptions on error
                                     // We don't throw on failbit, since we sometimes want to read
                                     // until the end of file in a loop of form `while
                                     // (std::getline(...))`, which will set failbit.
  return ifs;
}

inline std::ifstream OpenFileForReadAsStream(std::string const& filename) {
  return OpenFileForReadAsStream(std::filesystem::u8path(filename));
}

inline std::ifstream OpenFileForReadAsStream(char const* filename) {
  return OpenFileForReadAsStream(std::string(filename));
}

inline std::ofstream OpenFileForWriteAsStream(std::filesystem::path const& filepath) {
  auto path = std::filesystem::weakly_canonical(filepath);
  TREELITE_CHECK(path.has_filename()) << "Cannot write to a directory; please specify a file";
  TREELITE_CHECK(std::filesystem::exists(path.parent_path()))
      << "Path " << path.parent_path() << " does not exist";
  std::ofstream ofs(path, std::ios::out | std::ios::binary);
  TREELITE_CHECK(ofs) << "Could not open file " << filepath;
  ofs.exceptions(std::ios::failbit | std::ios::badbit);  // Throw exceptions on error
  return ofs;
}

inline std::ofstream OpenFileForWriteAsStream(std::string const& filename) {
  return OpenFileForWriteAsStream(std::filesystem::u8path(filename));
}

inline std::ofstream OpenFileForWriteAsStream(char const* filename) {
  return OpenFileForWriteAsStream(std::string(filename));
}

inline FILE* OpenFileForReadAsFilePtr(std::filesystem::path const& filepath) {
  auto path = std::filesystem::weakly_canonical(filepath);
  TREELITE_CHECK(std::filesystem::exists(path)) << "Path " << filepath << " does not exist";
  FILE* fp;
#ifdef _WIN32
  fp = _wfopen(path.wstring().c_str(), L"rb");
#else
  fp = std::fopen(path.string().c_str(), "rb");
#endif
  TREELITE_CHECK(fp) << "Could not open file " << filepath;
  return fp;
}

inline FILE* OpenFileForReadAsFilePtr(std::string const& filename) {
  return OpenFileForReadAsFilePtr(std::filesystem::u8path(filename));
}

inline FILE* OpenFileForReadAsFilePtr(char const* filename) {
  return OpenFileForReadAsFilePtr(std::string(filename));
}

}  // namespace treelite::detail

#endif  // TREELITE_DETAIL_FILE_UTILS_H_
