# frozen_string_literal: true

# @see https://en.wikipedia.org/wiki/Executable_and_Linkable_Format#File_header
module ELFShim
  MAGIC_NUMBER_OFFSET = 0
  MAGIC_NUMBER_ASCII = "\x7fELF"

  OS_ABI_OFFSET = 0x07
  OS_ABI_SYSTEM_V = 0
  OS_ABI_LINUX = 3

  TYPE_OFFSET = 0x10
  TYPE_EXECUTABLE = 2
  TYPE_SHARED = 3

  ARCHITECTURE_OFFSET = 0x12
  ARCHITECTURE_I386 = 0x3
  ARCHITECTURE_POWERPC = 0x14
  ARCHITECTURE_ARM = 0x28
  ARCHITECTURE_X86_64 = 0x62
  ARCHITECTURE_AARCH64 = 0xB7

  def read_uint8(offset)
    read(1, offset).unpack1("C")
  end

  def read_uint16(offset)
    read(2, offset).unpack1("v")
  end

  def elf?
    return @elf if defined? @elf
    return @elf = false unless read(MAGIC_NUMBER_ASCII.size, MAGIC_NUMBER_OFFSET) == MAGIC_NUMBER_ASCII

    # Check that this ELF file is for Linux or System V.
    # OS_ABI is often set to 0 (System V), regardless of the target platform.
    @elf = [OS_ABI_LINUX, OS_ABI_SYSTEM_V].include? read_uint8(OS_ABI_OFFSET)
  end

  def arch
    return :dunno unless elf?

    @arch ||= case read_uint16(ARCHITECTURE_OFFSET)
    when ARCHITECTURE_I386 then :i386
    when ARCHITECTURE_X86_64 then :x86_64
    when ARCHITECTURE_POWERPC then :powerpc
    when ARCHITECTURE_ARM then :arm
    when ARCHITECTURE_AARCH64 then :arm64
    else :dunno
    end
  end

  def elf_type
    return :dunno unless elf?

    @elf_type ||= case read_uint16(TYPE_OFFSET)
    when TYPE_EXECUTABLE then :executable
    when TYPE_SHARED then :dylib
    else :dunno
    end
  end

  def dylib?
    elf_type == :dylib
  end

  def binary_executable?
    elf_type == :executable
  end

  def rpath
    return @rpath if defined? @rpath

    @rpath = rpath_using_patchelf_rb
  end

  def interpreter
    return @interpreter if defined? @interpreter

    @interpreter = patchelf_patcher.interpreter
  end

  def dynamic_elf?
    return @dynamic_elf if defined? @dynamic_elf

    @dynamic_elf = patchelf_patcher.elf.segment_by_type(:DYNAMIC).present?
  end

  class Metadata
    attr_reader :path, :dylib_id, :dylibs

    def initialize(path)
      @path = path
      @dylibs = []
      @dylib_id, needed = needed_libraries path
      return if needed.empty?

      ldd = DevelopmentTools.locate "ldd"
      ldd_output = Utils.popen_read(ldd, path.expand_path.to_s).split("\n")
      return unless $CHILD_STATUS.success?

      ldd_paths = ldd_output.map do |line|
        match = line.match(/\t.+ => (.+) \(.+\)|\t(.+) => not found/)
        next unless match

        match.captures.compact.first
      end.compact
      @dylibs = ldd_paths.select do |ldd_path|
        next true unless ldd_path.start_with? "/"

        needed.include? File.basename(ldd_path)
      end
    end

    private

    def needed_libraries(path)
      return [nil, []] unless path.dynamic_elf?

      needed_libraries_using_patchelf_rb path
    end

    def needed_libraries_using_patchelf_rb(path)
      patcher = path.patchelf_patcher
      [patcher.soname, patcher.needed]
    end
  end

  def rpath_using_patchelf_rb
    patchelf_patcher.runpath || patchelf_patcher.rpath
  end

  def patchelf_patcher
    require "patchelf"
    @patchelf_patcher ||= PatchELF::Patcher.new to_s, on_error: :silent
  end

  def metadata
    @metadata ||= Metadata.new(self)
  end

  def dylib_id
    metadata.dylib_id
  end

  def dynamically_linked_libraries(*)
    metadata.dylibs
  end
end
