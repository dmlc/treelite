/*!
 * Copyright (c) 2023 by Contributors
 * \file test_utils.cc
 * \author Hyunsu Cho
 * \brief C++ tests for utility functions
 */

#include <gtest/gtest.h>
#include <treelite/detail/file_utils.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

TEST(FileUtils, StreamIO) {
  std::string s{"Hello world"};
  std::string s2;
  std::filesystem::path tmpdir = std::filesystem::temp_directory_path();
  std::filesystem::path filepath = tmpdir / std::filesystem::u8path("ななひら.txt");

  {
    std::ofstream ofs = treelite::detail::OpenFileForWriteAsStream(filepath);
    ofs.write(s.data(), s.length());
  }
  {
    std::ifstream ifs = treelite::detail::OpenFileForReadAsStream(filepath);
    s2.resize(s.length());
    ifs.read(s2.data(), s.length());
    ASSERT_EQ(s, s2);
  }

  std::filesystem::remove(filepath);
}

TEST(FileUtils, OpenFileForReadAsFilePtr) {
  std::string s{"Hello world"};
  std::string s2;
  std::filesystem::path tmpdir = std::filesystem::temp_directory_path();
  std::filesystem::path filepath = tmpdir / std::filesystem::u8path("ななひら.txt");

  {
    std::ofstream ofs(filepath, std::ios::out | std::ios::binary);
    ASSERT_TRUE(ofs);
    ofs.exceptions(std::ios::failbit | std::ios::badbit);
    ofs.write(s.data(), s.length());
  }
  {
    FILE* fp = treelite::detail::OpenFileForReadAsFilePtr(filepath);
    ASSERT_TRUE(fp);
    s2.resize(s.length());
    ASSERT_EQ(std::fread(s2.data(), sizeof(char), s.length(), fp), s.length());
    ASSERT_EQ(s, s2);
    ASSERT_EQ(std::fclose(fp), 0);
  }

  std::filesystem::remove(filepath);
}
