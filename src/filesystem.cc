/*!
 * Copyright (c) 2020-2021 by Contributors
 * \file filesystem.cc
 * \author Hyunsu Cho
 * \brief Cross-platform wrapper for common filesystem functions
 */

#include <treelite/filesystem.h>
#include <treelite/logging.h>
#include <fstream>

#ifdef _WIN32
#include <windows.h>
#include <Shlwapi.h>
#pragma comment(lib, "Shlwapi.lib")
#else
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <libgen.h>
#include <cstring>
#endif

namespace {

inline void HandleSystemError(const std::string& msg) {
#ifdef _WIN32
  LPVOID msg_buf;
  DWORD dw = GetLastError();
  FormatMessage(
    FORMAT_MESSAGE_ALLOCATE_BUFFER |
    FORMAT_MESSAGE_FROM_SYSTEM |
    FORMAT_MESSAGE_IGNORE_INSERTS,
    NULL,
    dw,
    MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
    (LPTSTR)&msg_buf,
    0, NULL);
  const std::string msg_err(static_cast<const char*>(msg_buf));
  LocalFree(msg_buf);
#else
  const std::string msg_err(strerror(errno));
#endif
  TREELITE_LOG(FATAL) << msg << "\nReason: " << msg_err;
}

}  // anonymous namespace

namespace treelite {
namespace filesystem {

void CreateDirectoryIfNotExist(const char* dirpath) {
#ifdef _WIN32
  DWORD ftyp = GetFileAttributesA(dirpath);
  if (ftyp == INVALID_FILE_ATTRIBUTES) {
    // directory doesn't seem to exist; attempt to create one
    if (CreateDirectoryA(dirpath, NULL) == 0) {
      // failed to create a new directory
      HandleSystemError(std::string("CreateDirectoryIfNotExist: "
                        "failed to create new directory ") + dirpath);
    }
  } else {
    if (!(ftyp & FILE_ATTRIBUTE_DIRECTORY)) {
      TREELITE_LOG(FATAL) << "CreateDirectoryIfNotExist: "
                          << dirpath << " is a file, not a directory";
    }
  }
#else
  struct stat sb;
  if (stat(dirpath, &sb) != 0) {
    // directory doesn't seem to exist; attempt to create one
    if (mkdir(dirpath, S_IRUSR | S_IWUSR | S_IXUSR) != 0) {
      // failed to create a new directory
      HandleSystemError(std::string("CreateDirectoryIfNotExist: "
                        "failed to create new directory ") + dirpath);
    }
  } else {
    if (!S_ISDIR(sb.st_mode)) {
      TREELITE_LOG(FATAL) << "CreateDirectoryIfNotExist: "
                          << dirpath << " is a file, not a directory";
    }
  }
#endif
}

void WriteToFile(const std::string& filename, const std::string& content) {
  std::ofstream of(filename);
  of << content;
}

void WriteToFile(const std::string& filename, const std::vector<char>& content) {
  std::ofstream of(filename, std::ios::out | std::ios::binary);
  of.write(content.data(), content.size());
}

}  // namespace filesystem
}  // namespace treelite
