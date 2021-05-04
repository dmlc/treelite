/*!
 * Copyright (c) 2021 by Contributors
 * \file serializer.cc
 * \brief Implementation for serialization related functions
 * \author Hyunsu Cho
 */

#include <treelite/tree.h>

namespace treelite {

void SerializePyBufferFrame(PyBufferFrame frame, FILE* dest_fp) {
  auto write_to_file = [](const void* buffer, std::size_t size, std::size_t count, FILE* fp) {
    if (std::fwrite(buffer, size, count, fp) < count) {
      throw std::runtime_error("Failed to write to disk");
    }
  };

  static_assert(sizeof(uint64_t) >= sizeof(std::size_t), "size_t too big on this platform");

  const auto itemsize_uint64 = static_cast<uint64_t>(frame.itemsize);
  const auto nitem_uint64 = static_cast<uint64_t>(frame.nitem);
  const auto format_str_len = static_cast<uint64_t>(std::strlen(frame.format));

  write_to_file(&itemsize_uint64, sizeof(itemsize_uint64), 1, dest_fp);
  write_to_file(&nitem_uint64, sizeof(nitem_uint64), 1, dest_fp);
  write_to_file(&format_str_len, sizeof(format_str_len), 1, dest_fp);
  write_to_file(frame.format, sizeof(char),
                static_cast<std::size_t>(format_str_len) + 1, dest_fp);  // write terminating NUL
  write_to_file(frame.buf, frame.itemsize, frame.nitem, dest_fp);
}

PyBufferFrame DeserializePyBufferFrame(
    FILE* src_fp, void** allocated_buf, char** allocated_format) {
  auto read_from_file = [](void* buffer, std::size_t size, std::size_t count, FILE* fp) {
    if (std::fread(buffer, size, count, fp) < count) {
      throw std::runtime_error("Failed to read from disk");
    }
  };
  auto alloc_error = []() {
    throw std::runtime_error("Failed to allocate buffer while deserializing");
  };

  uint64_t itemsize, nitem, format_str_len;
  void* buf;
  char* format;

  read_from_file(&itemsize, sizeof(itemsize), 1, src_fp);
  read_from_file(&nitem, sizeof(nitem), 1, src_fp);
  read_from_file(&format_str_len, sizeof(format_str_len), 1, src_fp);
  itemsize = static_cast<std::size_t>(itemsize);
  nitem = static_cast<std::size_t>(nitem);
  format = static_cast<char*>(std::malloc(
      sizeof(char) * (static_cast<std::size_t>(format_str_len) + 1)));
  if (!format) {
    alloc_error();
  }
  read_from_file(format, sizeof(char), static_cast<std::size_t>(format_str_len) + 1, src_fp);
  buf = static_cast<char*>(std::malloc(sizeof(char) * static_cast<std::size_t>(itemsize * nitem)));
  if (!buf) {
    alloc_error();
  }
  read_from_file(buf, itemsize, nitem, src_fp);

  if (allocated_buf) {
    *allocated_buf = buf;
  }
  if (allocated_format) {
    *allocated_format = format;
  }
  return PyBufferFrame{buf, format, itemsize, nitem};
}

void
Model::Serialize(FILE* dest_fp) {
  auto frames = this->GetPyBuffer();
  const auto num_frame = static_cast<uint64_t>(frames.size());
  if (std::fwrite(&num_frame, sizeof(num_frame), 1, dest_fp) < 1) {
    throw std::runtime_error("Error while serializing to disk");
  }
  for (auto frame : frames) {
    SerializePyBufferFrame(frame, dest_fp);
  }
}

std::unique_ptr<Model>
Model::Deserialize(FILE* src_fp) {
  uint64_t num_frame;
  if (std::fread(&num_frame, sizeof(num_frame), 1, src_fp) < 1) {
    throw std::runtime_error("Error while deserializing from disk");
  }
  std::vector<PyBufferFrame> frames;
  for (uint64_t i = 0; i < num_frame; ++i) {
    frames.push_back(DeserializePyBufferFrame(src_fp, nullptr, nullptr));
  }
  return CreateFromPyBufferImpl(frames, true);
  // Set assume_ownership=true to transfer the ownership of the two underlying buffers (buf,
  // format) of the frames to the Model object. All the buffers were allocated by
  // PyBufferFrame::Deserialize() and should be freed when the Model object is freed.
}

}  // namespace treelite
