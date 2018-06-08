/*!
 *  Copyright (c) 2017 by Contributors
 * \file filesystem.h
 * \author Philip Cho
 * \brief Cross-platform wrapper for common filesystem functions
 */
#ifndef TREELITE_COMMON_FILESYSTEM_H_
#define TREELITE_COMMON_FILESYSTEM_H_

#include <dmlc/logging.h>
#include <regex>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <cstdlib>
#else
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <libgen.h>
#include <cstring>
#endif

namespace treelite {
namespace common {
namespace filesystem {

/*!
 * \brief extract the base name from a full path. The base name is defined as
 *        the component that follows the last '/' in the full path.
 * \code
 *   GetBaseName("./food/bar.txt");  // returns bar.txt
 * \endcode
 * \param path full path
 */
inline std::string GetBasename(const std::string& path) {
#ifdef _WIN32
  /* remove any trailing backward or forward slashes
     (UNIX does this automatically) */
  std::string path_;
  std::string::size_type tmp = path.find_last_of("/\\");
  if (tmp == path.length() - 1) {
    size_t i = tmp;
    while ((path[i] == '/' || path[i] == '\\') && i >= 0) {
      --i;
    }
    path_ = path.substr(0, i + 1);
  } else {
    path_ = path;
  }
  std::vector<char> fname(path_.length() + 1);
  std::vector<char> ext(path_.length() + 1);
  _splitpath_s(path_.c_str(), NULL, 0, NULL, 0,
    &fname[0], path_.length() + 1, &ext[0], path_.length() + 1);
  return std::string(&fname[0]) + std::string(&ext[0]);
#else
  char* path_ = strdup(path.c_str());
  char* base = basename(path_);
  std::string ret(base);
  free(path_);
  return ret;
#endif
}

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
  LOG(FATAL) << msg << "\nReason: " << msg_err;
}

inline void CreateDirectoryIfNotExist(const char* dirpath) {
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
      LOG(FATAL) << "CreateDirectoryIfNotExist: "
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
      LOG(FATAL) << "CreateDirectoryIfNotExist: "
                 << dirpath << " is a file, not a directory";
    }
  }
#endif
}

inline void CreateDirectoryIfNotExistRecursive(const std::string& dirpath) {
  std::string dirpath_;
#ifdef _WIN32
  if (dirpath.find("/") == std::string::npos
      && dirpath.find("\\") != std::string::npos) {
    // replace backward slashes with forward slashes
    dirpath_ = std::regex_replace(dirpath, std::regex("\\\\"), "/");
  } else {
    dirpath_ = dirpath;
  }
#else
  dirpath_ = dirpath;
#endif
  const std::vector<std::string> tokens = common::Split(dirpath_, '/');
  std::string accum;
  size_t i;
  if (tokens[0].empty()) {  // absolute path, starting with '/'
    accum = "/" + tokens[1];
    i = 1;
  } else {  // relative path
    accum = tokens[0];
    i = 0;
  }
  for (; i < tokens.size(); ++i) {
    common::filesystem::CreateDirectoryIfNotExist(accum.c_str());
    if (i < tokens.size() - 1 && !tokens[i + 1].empty()) {
      accum += "/";
      accum += tokens[i + 1];
    }
  }
}

}  // namespace filesystem
}  // namespace common
}  // namespace treelite

#endif  // TREELITE_COMMON_FILESYSTEM_H_
