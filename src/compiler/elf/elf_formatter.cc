/*!
 * Copyright (c) 2019-2021 by Contributors
 * \file elf_formatter.cc
 * \author Hyunsu Cho
 * \brief Generate a relocatable object file containing a constant, read-only array
 */
#include <treelite/logging.h>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <vector>
#include <cstdio>
#include <cstring>
#include "./elf_formatter.h"

#ifdef __linux__

#include <elf.h>

namespace {

const unsigned int SHF_X86_64_LARGE = 0x10000000;

const char ident_str[EI_NIDENT] = {
  ELFMAG0, ELFMAG1, ELFMAG2, ELFMAG3,             // magic string: 0x7F, "ELF"
  ELFCLASS64, ELFDATA2LSB,                        // EI_CLASS, EI_DATA: 64-bit, little-endian
  EV_CURRENT,                                     // EI_VERSION: ELF version 1
  ELFOSABI_NONE,                                  // EI_OSABI: System V ABI or unspecified
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00  // EI_PAD: reserved
};

void AppendToBuffer(std::vector<char>* dest, const void* src, size_t count) {
  const size_t beg = dest->size();
  dest->resize(beg + count);
  std::memcpy(dest->data() + beg, src, count);
}

}   // anonymous namespace

namespace treelite {
namespace compiler {

void AllocateELFHeader(std::vector<char>* elf_buffer) {
  elf_buffer->resize(elf_buffer->size() + sizeof(Elf64_Ehdr));
}

void FormatArrayAsELF(std::vector<char>* elf_buffer) {
  const size_t array_size = elf_buffer->size() - sizeof(Elf64_Ehdr);

  /* Format compiler information string */
  const char comment[] = "\0GCC: (Ubuntu 7.4.0-1ubuntu1~18.04.1) 7.4.0\0\0\0\0";
    // padding added at the end so that the following section (.symtab) is 8-byte aligned.
  const size_t comment_padding = 4;  // remember how many NUL letters we added for padding
  static_assert(sizeof(comment) == 48, ".comment section has incorrect size");

  /* Format symbol table */
  const Elf64_Sym symtab[] = {
    // Each symbol entry is of form {st_name, st_info, st_other, st_shndx, st_value, st_size}
    // * st_name:  Symbol name. The symbol name is given by the null-terminated string that starts
    //             at &strtab[st_name].
    // * st_info:  Symbol's type and binding attributes
    // * st_other: Symbol visibility (we'll use STV_DEFAULT for all entries)
    // * st_shndx: Index of the section associated with the symbol
    //             (SHN_UNDEF: no section associated. SHN_ABS: index of file entry, by convention)
    // * st_value: Address associated with the symbol (we'll set this to 0 for all entries, since
    //             the object file is relocatable.)
    // * st_size:  Size (in bytes) of the symbol
    { 0, ELF64_ST_INFO(STB_LOCAL,  STT_NOTYPE), STV_DEFAULT, SHN_UNDEF, 0,          0},
    { 1, ELF64_ST_INFO(STB_LOCAL,    STT_FILE), STV_DEFAULT,   SHN_ABS, 0,          0},
    { 0, ELF64_ST_INFO(STB_LOCAL, STT_SECTION), STV_DEFAULT,         1, 0,          0},
    { 0, ELF64_ST_INFO(STB_LOCAL, STT_SECTION), STV_DEFAULT,         2, 0,          0},
    { 0, ELF64_ST_INFO(STB_LOCAL, STT_SECTION), STV_DEFAULT,         3, 0,          0},
    { 0, ELF64_ST_INFO(STB_LOCAL, STT_SECTION), STV_DEFAULT,         4, 0,          0},
    { 0, ELF64_ST_INFO(STB_LOCAL, STT_SECTION), STV_DEFAULT,         6, 0,          0},
    { 0, ELF64_ST_INFO(STB_LOCAL, STT_SECTION), STV_DEFAULT,         5, 0,          0},
    {10, ELF64_ST_INFO(STB_GLOBAL, STT_OBJECT), STV_DEFAULT,         4, 0, array_size},
  };
  static_assert(sizeof(symtab) == 216, ".symtab has incorrect size");

  /* Format symbol name table */
  const char strtab[] = "\0arrays.c\0nodes";
  static_assert(sizeof(strtab) == 16, ".strtab has incorrect size");

  /* Format section name table */
  const char shstrtab[] = "\0.symtab\0.strtab\0.shstrtab\0.text\0.data\0.bss\0.lrodata\0.comment\0"
                          ".note.GNU-stack\0\0";
    // padding added at the end to ensure 4-byte alignment everywhere
  const size_t shstrtab_padding = 2;  // remember how many NUL letters we added for padding
  static_assert(sizeof(shstrtab) == 80, ".shstrtab has incorrect size");

  /* Format ELF header */
  Elf64_Ehdr elf_header;
  // Compute e_shoff, section header table's offset.
  const size_t e_shoff = sizeof(elf_header) + array_size + sizeof(comment)
                         + sizeof(symtab) + sizeof(strtab) + sizeof(shstrtab);

  std::memcpy(elf_header.e_ident, ident_str, EI_NIDENT);
  elf_header.e_type = ET_REL;         // A relocatable (object) file
  elf_header.e_machine = EM_X86_64;   // AMD64 architecture target
  elf_header.e_version = EV_CURRENT;  // ELF version 1
  elf_header.e_entry = 0;             // Set to zero because there's no entry point
  elf_header.e_phoff = 0;             // Set to zero because there's no program header table
  elf_header.e_shoff = e_shoff;       // Section header table's offset
  elf_header.e_flags = 0;             // Reserved
  elf_header.e_ehsize = 64;           // Size of ELF header (in bytes)
  elf_header.e_phentsize = 0;         // Set to zero because there's no program header table
  elf_header.e_phnum = 0;             // Set to zero because there's no program header table
  elf_header.e_shentsize = 64;        // Size of each section header (in bytes)
  elf_header.e_shnum = 10;            // Number of section headers
  elf_header.e_shstrndx = 9;          // Index (in section header table) of the section storing
                                      // string representation of all section names
                                      // In this case, the last section stores name of all sections

  /* Format section header table */
  Elf64_Shdr section_header[] = {
    // Each section header is of form {sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size,
    //                                 sh_link, sh_info, sh_addralign, sh_entsize}
    // * sh_name:      Section name. The section name is given by the null-terminated string that
    //                 starts at &shstrtab[sh_name].
    // * sh_type:      Type of section
    // * sh_flags:     Miscellaneous attributes
    // * sh_addr:      Address of the first byte of the section (we'll set this to 0 for all
    //                 sections, since the object file is relocatable.)
    // * sh_offset:    Byte offset from the beginning of the object file to the first byte in the
    //                 section
    // * sh_size:      Size of section in bytes
    // * sh_link:      Interpretation of this field depends on the section type
    //                 See https://www.sco.com/developers/gabi/1998-04-29/ch4.sheader.html#sh_link
    // * sh_info:      Interpretation of this field depends on the section type
    //                 See https://www.sco.com/developers/gabi/1998-04-29/ch4.sheader.html#sh_link
    // * sh_addralign: Alignment constraint for the section. This must be a power of 2. A value of
    //                 0 or 1 indicates the lack of alignment constraint.
    // * sh_entsize:   Size of each entry in a table (in bytes). This is only applicable if the
    //                 section is a table of some kind (e.g. symbol table).
    { 0,     SHT_NULL,                          0x0, 0x0, 0x0,                0, 0, 0,  0,  0},
    {27, SHT_PROGBITS,    SHF_ALLOC | SHF_EXECINSTR, 0x0, 0x0,                0, 0, 0,  1,  0},
    {33, SHT_PROGBITS,        SHF_WRITE | SHF_ALLOC, 0x0, 0x0,                0, 0, 0,  1,  0},
    {39,   SHT_NOBITS,        SHF_WRITE | SHF_ALLOC, 0x0, 0x0,                0, 0, 0,  1,  0},
    {44, SHT_PROGBITS, SHF_ALLOC | SHF_X86_64_LARGE, 0x0, 0x0,       array_size, 0, 0, 32,  0},
    {53, SHT_PROGBITS,      SHF_MERGE | SHF_STRINGS, 0x0, 0x0,  sizeof(comment), 0, 0,  1,  1},
    {62, SHT_PROGBITS,                          0x0, 0x0, 0x0,                0, 0, 0,  1,  0},
    { 1,   SHT_SYMTAB,                          0x0, 0x0, 0x0,   sizeof(symtab), 8, 8,  8, 24},
    { 9,   SHT_STRTAB,                          0x0, 0x0, 0x0,   sizeof(strtab), 0, 0,  1,  0},
    {17,   SHT_STRTAB,                          0x0, 0x0, 0x0, sizeof(shstrtab), 0, 0,  1,  0}
    // Sections listed: (null)  .text     .data  .bss  .lrodata  .comment  .note.GNU-stack  .symtab
    //                  .strtab .shstrtab
    // Note that some sections are not actually present in the object (thus has size zero).
  };
  // Compute offsets via cumulative sums
  section_header[1].sh_offset = 0x40;
  for (size_t i = 2; i < sizeof(section_header) / sizeof(Elf64_Shdr); ++i) {
    section_header[i].sh_offset = section_header[i - 1].sh_offset + section_header[i - 1].sh_size;
  }
  // Adjust size info so that padding is excluded
  section_header[5].sh_size -= comment_padding;
  section_header[6].sh_offset -= comment_padding;
  section_header[9].sh_size -= shstrtab_padding;

  /**
   * Structure of ELF relocatable object file
   *
   * +---------------------------------+
   * | ELF Header                      |
   * +---------------------------------+
   * | .lrodata (read-only data) (*)   |
   * +---------------------------------+
   * | .comment (compiler information) |
   * +---------------------------------+
   * | .symtab (symbol table)          |
   * +---------------------------------+
   * | .strtab (symbol name table)     |
   * +---------------------------------+
   * | .shstrtab (section name table)  |
   * +---------------------------------+
   * | Section headers                 |
   * +---------------------------------+
   *
   *  (*) The 'l' prefix indicates that we enabled SHF_X86_64_LARGE flag for the data section, so
   *      that the section can hold more than 2 GB.
   *
   **/

  /* Write ELF header */
  std::memcpy(elf_buffer->data(), &elf_header, sizeof(Elf64_Ehdr));
    // elf_buffer already has a placeholder for the ELF header
  /* .lrodata (read-only data) segment is already part of elf_buffer */
  /* Write .comment (compiler information) segment */
  AppendToBuffer(elf_buffer, comment, sizeof(comment));
  /* Write .symtab (symbol table) segment */
  AppendToBuffer(elf_buffer, symtab, sizeof(symtab));
  /* Write .strtab (symbol name table) segment */
  AppendToBuffer(elf_buffer, strtab, sizeof(strtab));
  /* Write .shstrtab (section name table) segment (referred by elf_header.e_shstrndx) */
  AppendToBuffer(elf_buffer, shstrtab, sizeof(shstrtab));
  /* Write section headers */
  AppendToBuffer(elf_buffer, section_header, sizeof(section_header));
}

}  // namespace compiler
}  // namespace treelite

#else  // __linux__

namespace treelite {
namespace compiler {

void AllocateELFHeader(std::vector<char>* elf_buffer) {
  LOG(FATAL) << "dump_array_as_elf is not supported in non-Linux OSes";
}

void FormatArrayAsELF(std::vector<char>* elf_buffer) {
  LOG(FATAL) << "dump_array_as_elf is not supported in non-Linux OSes";
}

}  // namespace compiler
}  // namespace treelite

#endif  // __linux__
