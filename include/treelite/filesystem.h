/*!
 *  Copyright (c) 2020 by Contributors
 * \file filesystem.h
 * \author Hyunsu Cho
 * \brief Cross-platform wrapper for common filesystem functions
 */
#ifndef TREELITE_FILESYSTEM_H_
#define TREELITE_FILESYSTEM_H_

#include <string>
#include <vector>

namespace treelite {
namespace filesystem {

/*!
 * \brief Create a directory with a given name if one doesn't exist already.
 * \param dirpath Path to directory to be created.
 */
void CreateDirectoryIfNotExist(const char* dirpath);

/*!
 * \brief Write a sequence of strings to a text file, with newline character (\n) inserted between
 *        strings. This function is suitable for creating multi-line text files.
 * \param filename name of text file
 * \param lines a sequence of strings to be written.
 */
void WriteToFile(const std::string& filename, const std::string& content);

/*!
 * \brief Write a sequence of bytes to a text file
 * \param filename name of text file
 * \param lines a sequence of strings to be written.
 */
void WriteToFile(const std::string& filename, const std::vector<char>& content);

}  // namespace filesystem
}  // namespace treelite

#endif  // TREELITE_FILESYSTEM_H_
