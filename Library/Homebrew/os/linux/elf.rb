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

  def with_interpreter?
    return @with_interpreter if defined? @with_interpreter

    @with_interpreter = if binary_executable?
      true
    elsif dylib?
      if HOMEBREW_PATCHELF_RB
        begin
          patchelf_patcher.interpreter.present?
        rescue PatchELF::PatchError => e
          opoo e
          false
        end
      elsif which "readelf"
        Utils.popen_read("readelf", "-l", to_path).include?(" INTERP ")
      elsif which "file"
        Utils.popen_read("file", "-L", "-b", to_path).include?(" interpreter ")
      else
        raise "Please install either readelf (from binutils) or file."
      end
    else
      false
    end
  end

  def dynamic_elf?
    return @dynamic_elf if defined? @dynamic_elf

    @dynamic_elf = if HOMEBREW_PATCHELF_RB
      patchelf_patcher.instance_variable_get(:@elf).segment_by_type(:DYNAMIC).present?
    elsif which "readelf"
      Utils.popen_read("readelf", "-l", to_path).include?(" DYNAMIC ")
    elsif which "file"
      !Utils.popen_read("file", "-L", "-b", to_path)[/dynamic|shared/].nil?
    else
      raise "Please install either readelf (from binutils) or file."
    end
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
      if HOMEBREW_PATCHELF_RB
        needed_libraries_using_patchelf_rb path
      elsif DevelopmentTools.locate "readelf"
        needed_libraries_using_readelf path
      elsif DevelopmentTools.locate "patchelf"
        needed_libraries_using_patchelf path
      else
        return [nil, []] if path.basename.to_s == "patchelf"

        raise "patchelf must be installed: brew install patchelf"
      end
    end

    def needed_libraries_using_patchelf_rb(path)
      patcher = path.patchelf_patcher
      return [nil, []] unless patcher

      soname = begin
        patcher.soname
      rescue PatchELF::PatchError => e
        opoo e unless e.to_s.start_with? "Entry DT_SONAME not found, not a shared library?"
        nil
      end
      needed = begin
        patcher.needed
      rescue PatchELF::PatchError => e
        opoo e
        []
      end
      [soname, needed]
    end

    def needed_libraries_using_patchelf(path)
      return [nil, []] unless path.dynamic_elf?

      patchelf = DevelopmentTools.locate "patchelf"
      if path.dylib?
        command = [patchelf, "--print-soname", path.expand_path.to_s]
        soname = Utils.safe_popen_read(*command).chomp
      end
      command = [patchelf, "--print-needed", path.expand_path.to_s]
      needed = Utils.safe_popen_read(*command).split("\n")
      [soname, needed]
    end

    def needed_libraries_using_readelf(path)
      soname = nil
      needed = []
      command = ["readelf", "-d", path.expand_path.to_s]
      lines = Utils.popen_read(*command, err: :out).split("\n")
      lines.each do |s|
        next if s.start_with?("readelf: Warning: possibly corrupt ELF header")

        filename = s[/\[(.*)\]/, 1]
        next if filename.nil?

        if s.include? "(SONAME)"
          soname = filename
        elsif s.include? "(NEEDED)"
          needed << filename
        end
      end
      [soname, needed]
    end
  end

  def patchelf_patcher
    return unless HOMEBREW_PATCHELF_RB

    @patchelf_patcher ||= begin
      Homebrew.install_bundler_gems!
      require "patchelf"
      PatchELF::Patcher.new to_s, logging: false
    end
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
