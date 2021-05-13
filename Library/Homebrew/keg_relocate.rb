# typed: false
# frozen_string_literal: true

class Keg
  PREFIX_PLACEHOLDER = "@@HOMEBREW_PREFIX@@"
  CELLAR_PLACEHOLDER = "@@HOMEBREW_CELLAR@@"
  REPOSITORY_PLACEHOLDER = "@@HOMEBREW_REPOSITORY@@"
  LIBRARY_PLACEHOLDER = "@@HOMEBREW_LIBRARY@@"
  PERL_PLACEHOLDER = "@@HOMEBREW_PERL@@"

  class Relocation
    extend T::Sig

    RELOCATABLE_PATH_REGEX_PREFIX = /(?<![a-zA-Z0-9])/.freeze

    def initialize
      @replacement_map = {}
    end

    def freeze
      @replacement_map.freeze
      super
    end

    sig { params(key: Symbol, old_value: T.any(String, Regexp), new_value: String, path: T::Boolean).void }
    def add_replacement_pair(key, old_value, new_value, path: false)
      old_value = self.class.path_to_regex(old_value) if path
      @replacement_map[key] = [old_value, new_value]
    end

    sig { params(key: Symbol).returns(T::Array[T.any(String, Regexp)]) }
    def replacement_pair_for(key)
      @replacement_map.fetch(key)
    end

    sig { params(text: String).void }
    def replace_text(text)
      replacements = @replacement_map.values.to_h

      sorted_keys = replacements.keys.sort_by do |key|
        key.is_a?(String) ? key.length : 999
      end.reverse

      any_changed = false
      sorted_keys.each do |key|
        changed = text.gsub!(key, replacements[key])
        any_changed ||= changed
      end
      any_changed
    end

    sig { params(path: T.any(String, Regexp)).returns(Regexp) }
    def self.path_to_regex(path)
      path = case path
      when String
        Regexp.escape(path)
      when Regexp
        path.source
      end
      Regexp.new(RELOCATABLE_PATH_REGEX_PREFIX.source + path)
    end
  end

  def fix_dynamic_linkage
    symlink_files.each do |file|
      link = file.readlink
      # Don't fix relative symlinks
      next unless link.absolute?

      link_starts_cellar = link.to_s.start_with?(HOMEBREW_CELLAR.to_s)
      link_starts_prefix = link.to_s.start_with?(HOMEBREW_PREFIX.to_s)
      next if !link_starts_cellar && !link_starts_prefix

      new_src = link.relative_path_from(file.parent)
      file.unlink
      FileUtils.ln_s(new_src, file)
    end
  end
  alias generic_fix_dynamic_linkage fix_dynamic_linkage

  def relocate_dynamic_linkage(_relocation)
    []
  end

  def prepare_relocation_to_placeholders
    relocation = Relocation.new
    relocation.add_replacement_pair(:prefix, HOMEBREW_PREFIX.to_s, PREFIX_PLACEHOLDER, path: true)
    relocation.add_replacement_pair(:cellar, HOMEBREW_CELLAR.to_s, CELLAR_PLACEHOLDER, path: true)
    # when HOMEBREW_PREFIX == HOMEBREW_REPOSITORY we should use HOMEBREW_PREFIX for all relocations to avoid
    # being unable to differentiate between them.
    if HOMEBREW_PREFIX != HOMEBREW_REPOSITORY
      relocation.add_replacement_pair(:repository, HOMEBREW_REPOSITORY.to_s, REPOSITORY_PLACEHOLDER, path: true)
    end
    relocation.add_replacement_pair(:library, HOMEBREW_LIBRARY.to_s, LIBRARY_PLACEHOLDER, path: true)
    relocation.add_replacement_pair(:perl,
                                    %r{\A#!(?:/usr/bin/perl\d\.\d+|#{HOMEBREW_PREFIX}/opt/perl/bin/perl)( |$)}o,
                                    "#!#{PERL_PLACEHOLDER}\\1")
    relocation
  end
  alias generic_prepare_relocation_to_placeholders prepare_relocation_to_placeholders

  def replace_locations_with_placeholders
    relocation = prepare_relocation_to_placeholders.freeze
    relocate_dynamic_linkage(relocation)
    replace_text_in_files(relocation)
  end

  def prepare_relocation_to_locations
    relocation = Relocation.new
    relocation.add_replacement_pair(:prefix, PREFIX_PLACEHOLDER, HOMEBREW_PREFIX.to_s)
    relocation.add_replacement_pair(:cellar, CELLAR_PLACEHOLDER, HOMEBREW_CELLAR.to_s)
    relocation.add_replacement_pair(:repository, REPOSITORY_PLACEHOLDER, HOMEBREW_REPOSITORY.to_s)
    relocation.add_replacement_pair(:library, LIBRARY_PLACEHOLDER, HOMEBREW_LIBRARY.to_s)
    relocation.add_replacement_pair(:perl, PERL_PLACEHOLDER, "#{HOMEBREW_PREFIX}/opt/perl/bin/perl")
    relocation
  end
  alias generic_prepare_relocation_to_locations prepare_relocation_to_locations

  def replace_placeholders_with_locations(files, skip_linkage: false)
    relocation = prepare_relocation_to_locations.freeze
    relocate_dynamic_linkage(relocation) unless skip_linkage
    replace_text_in_files(relocation, files: files)
  end

  def replace_text_in_files(relocation, files: nil)
    files ||= text_files | libtool_files

    changed_files = []
    files.map(&path.method(:join)).group_by { |f| f.stat.ino }.each_value do |first, *rest|
      s = first.open("rb", &:read)

      next unless relocation.replace_text(s)

      changed_files += [first, *rest].map { |file| file.relative_path_from(path) }

      begin
        first.atomic_write(s)
      rescue SystemCallError
        first.ensure_writable do
          first.open("wb") { |f| f.write(s) }
        end
      else
        rest.each { |file| FileUtils.ln(first, file, force: true) }
      end
    end
    changed_files
  end

  def detect_cxx_stdlibs(_options = {})
    []
  end

  def recursive_fgrep_args
    # for GNU grep; overridden for BSD grep on OS X
    "-lr"
  end
  alias generic_recursive_fgrep_args recursive_fgrep_args

  def each_unique_file_matching(string)
    Utils.popen_read("fgrep", recursive_fgrep_args, string, to_s) do |io|
      hardlinks = Set.new

      until io.eof?
        file = Pathname.new(io.readline.chomp)
        next if file.symlink?

        yield file if hardlinks.add? file.stat.ino
      end
    end
  end

  def lib
    path/"lib"
  end

  def libexec
    path/"libexec"
  end

  def text_files
    text_files = []
    return text_files if !which("file") || !which("xargs")

    # file has known issues with reading files on other locales. Has
    # been fixed upstream for some time, but a sufficiently new enough
    # file with that fix is only available in macOS Sierra.
    # https://bugs.gw.com/view.php?id=292
    with_custom_locale("C") do
      files = Set.new path.find.reject { |pn|
        next true if pn.symlink?
        next true if pn.directory?
        next false if pn.basename.to_s == "orig-prefix.txt" # for python virtualenvs
        next true if pn == self/".brew/#{name}.rb"
        next true if Metafiles::EXTENSIONS.include?(pn.extname)

        if pn.text_executable?
          text_files << pn
          next true
        end
        false
      }
      output, _status = Open3.capture2("xargs -0 file --no-dereference --print0",
                                       stdin_data: files.to_a.join("\0"))
      # `file` output sometimes contains data from the file, which may include
      # invalid UTF-8 entities, so tell Ruby this is just a bytestring
      output.force_encoding(Encoding::ASCII_8BIT)
      output.each_line do |line|
        path, info = line.split("\0", 2)
        # `file` sometimes prints more than one line of output per file;
        # subsequent lines do not contain a null-byte separator, so `info`
        # will be `nil` for those lines
        next unless info
        next unless info.include?("text")

        path = Pathname.new(path)
        next unless files.include?(path)

        text_files << path
      end
    end

    text_files
  end

  def libtool_files
    libtool_files = []

    path.find do |pn|
      next if pn.symlink? || pn.directory? || Keg::LIBTOOL_EXTENSIONS.exclude?(pn.extname)

      libtool_files << pn
    end
    libtool_files
  end

  def symlink_files
    symlink_files = []
    path.find do |pn|
      symlink_files << pn if pn.symlink?
    end

    symlink_files
  end

  def self.text_matches_in_file(file, string, ignores, linked_libraries, formula_and_runtime_deps_names)
    text_matches = []
    path_regex = Relocation.path_to_regex(string)
    Utils.popen_read("strings", "-t", "x", "-", file.to_s) do |io|
      until io.eof?
        str = io.readline.chomp
        next if ignores.any? { |i| i =~ str }
        next unless str.match? path_regex

        offset, match = str.split(" ", 2)

        # Some binaries contain strings with lists of files
        # e.g. `/usr/local/lib/foo:/usr/local/share/foo:/usr/lib/foo`
        # Each item in the list should be checked separately
        match.split(":").each do |sub_match|
          # Not all items in the list may be matches
          next unless sub_match.match? path_regex
          next if linked_libraries.include? sub_match # Don't bother reporting a string if it was found by otool

          # Do not report matches to files that do not exist.
          next unless File.exist? sub_match

          # Do not report matches to build dependencies.
          if formula_and_runtime_deps_names.present?
            begin
              keg_name = Keg.for(Pathname.new(sub_match)).name
              next unless formula_and_runtime_deps_names.include? keg_name
            rescue NotAKegError
              nil
            end
          end

          text_matches << [match, offset] unless text_matches.any? { |text| text.last == offset }
        end
      end
    end
    text_matches
  end

  def self.file_linked_libraries(_file, _string)
    []
  end

  def self.relocation_formulae
    []
  end

  def self.bottle_dependencies
    relocation_formulae
  end
end

require "extend/os/keg_relocate"
