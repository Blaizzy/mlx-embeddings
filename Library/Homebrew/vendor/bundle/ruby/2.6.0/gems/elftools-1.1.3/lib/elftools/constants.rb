# frozen_string_literal: true

module ELFTools
  # Define constants from elf.h.
  # Mostly refer from https://github.com/torvalds/linux/blob/master/include/uapi/linux/elf.h
  # and binutils/elfcpp/elfcpp.h.
  module Constants
    # ELF magic header
    ELFMAG = "\x7FELF"

    # Values of `d_un.d_val' in the DT_FLAGS and DT_FLAGS_1 entry.
    module DF
      DF_ORIGIN       = 0x00000001 # Object may use DF_ORIGIN
      DF_SYMBOLIC     = 0x00000002 # Symbol resolutions starts here
      DF_TEXTREL      = 0x00000004 # Object contains text relocations
      DF_BIND_NOW     = 0x00000008 # No lazy binding for this object
      DF_STATIC_TLS   = 0x00000010 # Module uses the static TLS model

      DF_1_NOW        = 0x00000001 # Set RTLD_NOW for this object.
      DF_1_GLOBAL     = 0x00000002 # Set RTLD_GLOBAL for this object.
      DF_1_GROUP      = 0x00000004 # Set RTLD_GROUP for this object.
      DF_1_NODELETE   = 0x00000008 # Set RTLD_NODELETE for this object.
      DF_1_LOADFLTR   = 0x00000010 # Trigger filtee loading at runtime.
      DF_1_INITFIRST  = 0x00000020 # Set RTLD_INITFIRST for this object
      DF_1_NOOPEN     = 0x00000040 # Set RTLD_NOOPEN for this object.
      DF_1_ORIGIN     = 0x00000080 # $ORIGIN must be handled.
      DF_1_DIRECT     = 0x00000100 # Direct binding enabled.
      DF_1_TRANS      = 0x00000200 # :nodoc:
      DF_1_INTERPOSE  = 0x00000400 # Object is used to interpose.
      DF_1_NODEFLIB   = 0x00000800 # Ignore default lib search path.
      DF_1_NODUMP     = 0x00001000 # Object can't be dldump'ed.
      DF_1_CONFALT    = 0x00002000 # Configuration alternative created.
      DF_1_ENDFILTEE  = 0x00004000 # Filtee terminates filters search.
      DF_1_DISPRELDNE = 0x00008000 # Disp reloc applied at build time.
      DF_1_DISPRELPND = 0x00010000 # Disp reloc applied at run-time.
      DF_1_NODIRECT   = 0x00020000 # Object has no-direct binding.
      DF_1_IGNMULDEF  = 0x00040000 # :nodoc:
      DF_1_NOKSYMS    = 0x00080000 # :nodoc:
      DF_1_NOHDR      = 0x00100000 # :nodoc:
      DF_1_EDITED     = 0x00200000 # Object is modified after built.
      DF_1_NORELOC    = 0x00400000 # :nodoc:
      DF_1_SYMINTPOSE = 0x00800000 # Object has individual interposers.
      DF_1_GLOBAUDIT  = 0x01000000 # Global auditing required.
      DF_1_SINGLETON  = 0x02000000 # Singleton symbols are used.
    end
    include DF

    # Dynamic table types, records in +d_tag+.
    module DT
      DT_NULL         = 0 # marks the end of the _DYNAMIC array
      DT_NEEDED       = 1 # libraries need to be linked by loader
      DT_PLTRELSZ     = 2 # total size of relocation entries
      DT_PLTGOT       = 3 # address of procedure linkage table or global offset table
      DT_HASH         = 4 # address of symbol hash table
      DT_STRTAB       = 5 # address of string table
      DT_SYMTAB       = 6 # address of symbol table
      DT_RELA         = 7 # address of a relocation table
      DT_RELASZ       = 8 # total size of the {DT_RELA} table
      DT_RELAENT      = 9 # size of each entry in the {DT_RELA} table
      DT_STRSZ        = 10 # total size of {DT_STRTAB}
      DT_SYMENT       = 11 # size of each entry in {DT_SYMTAB}
      DT_INIT         = 12 # where the initialization function is
      DT_FINI         = 13 # where the termination function is
      DT_SONAME       = 14 # the shared object name
      DT_RPATH        = 15 # has been superseded by {DT_RUNPATH}
      DT_SYMBOLIC     = 16 # has been superseded by the DF_SYMBOLIC flag
      DT_REL          = 17 # similar to {DT_RELA}
      DT_RELSZ        = 18 # total size of the {DT_REL} table
      DT_RELENT       = 19 # size of each entry in the {DT_REL} table
      DT_PLTREL       = 20 # type of relocation entry, either {DT_REL} or {DT_RELA}
      DT_DEBUG        = 21 # for debugging
      DT_TEXTREL      = 22 # has been superseded by the DF_TEXTREL flag
      DT_JMPREL       = 23 # address of relocation entries that are associated solely with the procedure linkage table
      DT_BIND_NOW     = 24 # if the loader needs to do relocate now, superseded by the DF_BIND_NOW flag
      DT_INIT_ARRAY   = 25 # address init array
      DT_FINI_ARRAY   = 26 # address of fini array
      DT_INIT_ARRAYSZ = 27 # total size of init array
      DT_FINI_ARRAYSZ = 28 # total size of fini array
      DT_RUNPATH      = 29 # path of libraries for searching
      DT_FLAGS        = 30 # flags
      DT_ENCODING     = 32 # just a lower bound
      # Values between {DT_LOOS} and {DT_HIOS} are reserved for operating system-specific semantics.
      DT_LOOS         = 0x6000000d
      DT_HIOS         = 0x6ffff000 # see {DT_LOOS}
      # Values between {DT_VALRNGLO} and {DT_VALRNGHI} use the +d_un.d_val+ field of the dynamic structure.
      DT_VALRNGLO     = 0x6ffffd00
      DT_VALRNGHI     = 0x6ffffdff # see {DT_VALRNGLO}
      # Values between {DT_ADDRRNGLO} and {DT_ADDRRNGHI} use the +d_un.d_ptr+ field of the dynamic structure.
      DT_ADDRRNGLO    = 0x6ffffe00
      DT_GNU_HASH     = 0x6ffffef5 # the gnu hash
      DT_ADDRRNGHI    = 0x6ffffeff # see {DT_ADDRRNGLO}
      DT_RELACOUNT    = 0x6ffffff9 # relative relocation count
      DT_RELCOUNT     = 0x6ffffffa # relative relocation count
      DT_FLAGS_1      = 0x6ffffffb # flags
      DT_VERDEF       = 0x6ffffffc # address of version definition table
      DT_VERDEFNUM    = 0x6ffffffd # number of entries in {DT_VERDEF}
      DT_VERNEED      = 0x6ffffffe # address of version dependency table
      DT_VERSYM       = 0x6ffffff0 # section address of .gnu.version
      DT_VERNEEDNUM   = 0x6fffffff # number of entries in {DT_VERNEED}
      # Values between {DT_LOPROC} and {DT_HIPROC} are reserved for processor-specific semantics.
      DT_LOPROC       = 0x70000000
      DT_HIPROC       = 0x7fffffff # see {DT_LOPROC}
    end
    include DT

    # These constants define the various ELF target machines.
    module EM
      EM_NONE           = 0      # none
      EM_M32            = 1      # AT&T WE 32100
      EM_SPARC          = 2      # SPARC
      EM_386            = 3      # Intel 80386
      EM_68K            = 4      # Motorola 68000
      EM_88K            = 5      # Motorola 88000
      EM_486            = 6      # Intel 80486
      EM_860            = 7      # Intel 80860
      EM_MIPS           = 8      # MIPS R3000 (officially, big-endian only)

      # Next two are historical and binaries and
      # modules of these types will be rejected by Linux.
      EM_MIPS_RS3_LE    = 10     # MIPS R3000 little-endian
      EM_MIPS_RS4_BE    = 10     # MIPS R4000 big-endian

      EM_PARISC         = 15     # HPPA
      EM_SPARC32PLUS    = 18     # Sun's "v8plus"
      EM_PPC            = 20     # PowerPC
      EM_PPC64          = 21     # PowerPC64
      EM_SPU            = 23     # Cell BE SPU
      EM_ARM            = 40     # ARM 32 bit
      EM_SH             = 42     # SuperH
      EM_SPARCV9        = 43     # SPARC v9 64-bit
      EM_H8_300         = 46     # Renesas H8/300
      EM_IA_64          = 50     # HP/Intel IA-64
      EM_X86_64         = 62     # AMD x86-64
      EM_S390           = 22     # IBM S/390
      EM_CRIS           = 76     # Axis Communications 32-bit embedded processor
      EM_M32R           = 88     # Renesas M32R
      EM_MN10300        = 89     # Panasonic/MEI MN10300, AM33
      EM_OPENRISC       = 92     # OpenRISC 32-bit embedded processor
      EM_BLACKFIN       = 106    # ADI Blackfin Processor
      EM_ALTERA_NIOS2   = 113    # Altera Nios II soft-core processor
      EM_TI_C6000       = 140    # TI C6X DSPs
      EM_AARCH64        = 183    # ARM 64 bit
      EM_TILEPRO        = 188    # Tilera TILEPro
      EM_MICROBLAZE     = 189    # Xilinx MicroBlaze
      EM_TILEGX         = 191    # Tilera TILE-Gx
      EM_BPF            = 247    # Linux BPF - in-kernel virtual machine
      EM_FRV            = 0x5441 # Fujitsu FR-V
      EM_AVR32          = 0x18ad # Atmel AVR32

      # This is an interim value that we will use until the committee comes up with a final number.
      EM_ALPHA          = 0x9026

      # Bogus old m32r magic number, used by old tools.
      EM_CYGNUS_M32R    = 0x9041
      # This is the old interim value for S/390 architecture
      EM_S390_OLD       = 0xA390
      # Also Panasonic/MEI MN10300, AM33
      EM_CYGNUS_MN10300 = 0xbeef

      # Return the architecture name according to +val+.
      # Used by {ELFTools::ELFFile#machine}.
      #
      # Only supports famous archs.
      # @param [Integer] val Value of +e_machine+.
      # @return [String]
      #   Name of architecture.
      # @example
      #   mapping(3)
      #   #=> 'Intel 80386'
      #   mapping(6)
      #   #=> 'Intel 80386'
      #   mapping(62)
      #   #=> 'Advanced Micro Devices X86-64'
      #   mapping(1337)
      #   #=> '<unknown>: 0x539'
      def self.mapping(val)
        case val
        when EM_NONE then 'None'
        when EM_386, EM_486 then 'Intel 80386'
        when EM_860 then 'Intel 80860'
        when EM_MIPS then 'MIPS R3000'
        when EM_PPC then 'PowerPC'
        when EM_PPC64 then 'PowerPC64'
        when EM_ARM then 'ARM'
        when EM_IA_64 then 'Intel IA-64'
        when EM_AARCH64 then 'AArch64'
        when EM_X86_64 then 'Advanced Micro Devices X86-64'
        else format('<unknown>: 0x%x', val)
        end
      end
    end
    include EM

    # This module defines elf file types.
    module ET
      ET_NONE = 0 # no file type
      ET_REL  = 1 # relocatable file
      ET_EXEC = 2 # executable file
      ET_DYN  = 3 # shared object
      ET_CORE = 4 # core file
      # Return the type name according to +e_type+ in ELF file header.
      # @return [String] Type in string format.
      def self.mapping(type)
        case type
        when Constants::ET_NONE then 'NONE'
        when Constants::ET_REL then 'REL'
        when Constants::ET_EXEC then 'EXEC'
        when Constants::ET_DYN then 'DYN'
        when Constants::ET_CORE then 'CORE'
        else '<unknown>'
        end
      end
    end
    include ET

    # Program header permission flags, records bitwise OR value in +p_flags+.
    module PF
      PF_X = 1
      PF_W = 2
      PF_R = 4
    end
    include PF

    # Program header types, records in +p_type+.
    module PT
      PT_NULL         = 0          # null segment
      PT_LOAD         = 1          # segment to be load
      PT_DYNAMIC      = 2          # dynamic tags
      PT_INTERP       = 3          # interpreter, same as .interp section
      PT_NOTE         = 4          # same as .note* section
      PT_SHLIB        = 5          # reserved
      PT_PHDR         = 6          # where program header starts
      PT_TLS          = 7          # thread local storage segment
      PT_LOOS         = 0x60000000 # OS-specific
      PT_HIOS         = 0x6fffffff # OS-specific
      # Values between {PT_LOPROC} and {PT_HIPROC} are reserved for processor-specific semantics.
      PT_LOPROC       = 0x70000000
      PT_HIPROC       = 0x7fffffff # see {PT_LOPROC}
      PT_GNU_EH_FRAME = 0x6474e550 # for exception handler
      PT_GNU_STACK    = 0x6474e551 # permission of stack
      PT_GNU_RELRO    = 0x6474e552 # read only after relocation
    end
    include PT

    # Special indices to section. These are used when there is no valid index to section header.
    # The meaning of these values is left upto the embedding header.
    module SHN
      SHN_UNDEF     = 0      # undefined section
      SHN_LORESERVE = 0xff00 # start of reserved indices
    end
    include SHN

    # Section header types, records in +sh_type+.
    module SHT
      SHT_NULL     = 0 # null section
      SHT_PROGBITS = 1 # information defined by program itself
      SHT_SYMTAB   = 2 # symbol table section
      SHT_STRTAB   = 3 # string table section
      SHT_RELA     = 4 # relocation with addends
      SHT_HASH     = 5 # symbol hash table
      SHT_DYNAMIC  = 6 # information of dynamic linking
      SHT_NOTE     = 7 # note section
      SHT_NOBITS   = 8 # section occupies no space
      SHT_REL      = 9 # relocation
      SHT_SHLIB    = 10 # reserved
      SHT_DYNSYM   = 11 # symbols for dynamic
      # Values between {SHT_LOPROC} and {SHT_HIPROC} are reserved for processor-specific semantics.
      SHT_LOPROC   = 0x70000000
      SHT_HIPROC   = 0x7fffffff # see {SHT_LOPROC}
      # Values between {SHT_LOUSER} and {SHT_HIUSER} are reserved for application programs.
      SHT_LOUSER   = 0x80000000
      SHT_HIUSER   = 0xffffffff # see {SHT_LOUSER}
    end
    include SHT

    # Symbol binding from Sym st_info field.
    module STB
      STB_LOCAL      = 0 # Local symbol
      STB_GLOBAL     = 1 # Global symbol
      STB_WEAK       = 2 # Weak symbol
      STB_NUM        = 3 # Number of defined types.
      STB_LOOS       = 10 # Start of OS-specific
      STB_GNU_UNIQUE = 10 # Unique symbol.
      STB_HIOS       = 12 # End of OS-specific
      STB_LOPROC     = 13 # Start of processor-specific
      STB_HIPROC     = 15 # End of processor-specific
    end
    include STB

    # Symbol types from Sym st_info field.
    module STT
      STT_NOTYPE         = 0 # Symbol type is unspecified
      STT_OBJECT         = 1 # Symbol is a data object
      STT_FUNC           = 2 # Symbol is a code object
      STT_SECTION        = 3 # Symbol associated with a section
      STT_FILE           = 4 # Symbol's name is file name
      STT_COMMON         = 5 # Symbol is a common data object
      STT_TLS            = 6 # Symbol is thread-local data object
      STT_NUM            = 7 # Number of defined types.

      # GNU extension: symbol value points to a function which is called
      # at runtime to determine the final value of the symbol.
      STT_GNU_IFUNC      = 10

      STT_LOOS           = 10 # Start of OS-specific
      STT_HIOS           = 12 # End of OS-specific
      STT_LOPROC         = 13 # Start of processor-specific
      STT_HIPROC         = 15 # End of processor-specific

      # The section type that must be used for register symbols on
      # Sparc. These symbols initialize a global register.
      STT_SPARC_REGISTER = 13

      # ARM: a THUMB function. This is not defined in ARM ELF Specification but
      # used by the GNU tool-chain.
      STT_ARM_TFUNC      = 13
    end
    include STT
  end
end
