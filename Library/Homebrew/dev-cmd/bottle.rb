# typed: false
# frozen_string_literal: true

require "formula"
require "utils/bottles"
require "tab"
require "keg"
require "formula_versions"
require "cli/parser"
require "utils/inreplace"
require "erb"
require "archive"
require "bintray"

BOTTLE_ERB = <<-EOS
  bottle do
    <% if [HOMEBREW_BOTTLE_DEFAULT_DOMAIN.to_s,
           "#{HOMEBREW_BOTTLE_DEFAULT_DOMAIN}/bottles"].exclude?(root_url) %>
    root_url "<%= root_url %>"
    <% end %>
    <% if rebuild.positive? %>
    rebuild <%= rebuild %>
    <% end %>
    <% sha256_lines.each do |line| %>
    <%= line %>
    <% end %>
  end
EOS

MAXIMUM_STRING_MATCHES = 100

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def bottle_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Generate a bottle (binary package) from a formula that was installed with
        `--build-bottle`.
        If the formula specifies a rebuild version, it will be incremented in the
        generated DSL. Passing `--keep-old` will attempt to keep it at its original
        value, while `--no-rebuild` will remove it.
      EOS
      switch "--skip-relocation",
             description: "Do not check if the bottle can be marked as relocatable."
      switch "--force-core-tap",
             description: "Build a bottle even if <formula> is not in `homebrew/core` or any installed taps."
      switch "--no-rebuild",
             description: "If the formula specifies a rebuild version, remove it from the generated DSL."
      switch "--keep-old",
             description: "If the formula specifies a rebuild version, attempt to preserve its value in the "\
                          "generated DSL."
      switch "--json",
             description: "Write bottle information to a JSON file, which can be used as the value for "\
                          "`--merge`."
      switch "--merge",
             description: "Generate an updated bottle block for a formula and optionally merge it into the "\
                          "formula file. Instead of a formula name, requires the path to a JSON file generated with "\
                          "`brew bottle --json` <formula>."
      switch "--write",
             depends_on:  "--merge",
             description: "Write changes to the formula file. A new commit will be generated unless "\
                          "`--no-commit` is passed."
      switch "--no-commit",
             depends_on:  "--write",
             description: "When passed with `--write`, a new commit will not generated after writing changes "\
                          "to the formula file."
      switch "--only-json-tab",
             depends_on:  "--json",
             description: "When passed with `--json`, the tab will be written to the JSON file but not the bottle."
      flag   "--committer=",
             description: "Specify a committer name and email in `git`'s standard author format."
      flag   "--root-url=",
             description: "Use the specified <URL> as the root of the bottle's URL instead of Homebrew's default."

      conflicts "--no-rebuild", "--keep-old"

      named_args [:installed_formula, :file], min: 1
    end
  end

  def bottle
    args = bottle_args.parse

    return merge(args: args) if args.merge?

    ensure_relocation_formulae_installed! unless args.skip_relocation?
    args.named.to_resolved_formulae(uniq: false).each do |f|
      bottle_formula f, args: args
    end
  end

  def ensure_relocation_formulae_installed!
    Keg.relocation_formulae.each do |f|
      next if Formula[f].latest_version_installed?

      ohai "Installing #{f}..."
      safe_system HOMEBREW_BREW_FILE, "install", f
    end
  end

  def keg_contain?(string, keg, ignores, formula_and_runtime_deps_names = nil, args:)
    @put_string_exists_header, @put_filenames = nil

    print_filename = lambda do |str, filename|
      unless @put_string_exists_header
        opoo "String '#{str}' still exists in these files:"
        @put_string_exists_header = true
      end

      @put_filenames ||= []

      return if @put_filenames.include?(filename)

      puts Formatter.error(filename.to_s)
      @put_filenames << filename
    end

    result = false

    keg.each_unique_file_matching(string) do |file|
      next if Metafiles::EXTENSIONS.include?(file.extname) # Skip document files.

      linked_libraries = Keg.file_linked_libraries(file, string)
      result ||= !linked_libraries.empty?

      if args.verbose?
        print_filename.call(string, file) unless linked_libraries.empty?
        linked_libraries.each do |lib|
          puts " #{Tty.bold}-->#{Tty.reset} links to #{lib}"
        end
      end

      text_matches = []

      # Use strings to search through the file for each string
      Utils.popen_read("strings", "-t", "x", "-", file.to_s) do |io|
        until io.eof?
          str = io.readline.chomp
          next if ignores.any? { |i| i =~ str }
          next unless str.include? string

          offset, match = str.split(" ", 2)
          next if linked_libraries.include? match # Don't bother reporting a string if it was found by otool

          # Do not report matches to files that do not exist.
          next unless File.exist? match

          # Do not report matches to build dependencies.
          if formula_and_runtime_deps_names.present?
            begin
              keg_name = Keg.for(Pathname.new(match)).name
              next unless formula_and_runtime_deps_names.include? keg_name
            rescue NotAKegError
              nil
            end
          end

          result = true
          text_matches << [match, offset]
        end
      end

      next if !args.verbose? || text_matches.empty?

      print_filename.call(string, file)
      text_matches.first(MAXIMUM_STRING_MATCHES).each do |match, offset|
        puts " #{Tty.bold}-->#{Tty.reset} match '#{match}' at offset #{Tty.bold}0x#{offset}#{Tty.reset}"
      end

      if text_matches.size > MAXIMUM_STRING_MATCHES
        puts "Only the first #{MAXIMUM_STRING_MATCHES} matches were output."
      end
    end

    keg_contain_absolute_symlink_starting_with?(string, keg, args: args) || result
  end

  def keg_contain_absolute_symlink_starting_with?(string, keg, args:)
    absolute_symlinks_start_with_string = []
    keg.find do |pn|
      next if !pn.symlink? || !(link = pn.readlink).absolute?

      absolute_symlinks_start_with_string << pn if link.to_s.start_with?(string)
    end

    if args.verbose? && absolute_symlinks_start_with_string.present?
      opoo "Absolute symlink starting with #{string}:"
      absolute_symlinks_start_with_string.each do |pn|
        puts "  #{pn} -> #{pn.resolved_path}"
      end
    end

    !absolute_symlinks_start_with_string.empty?
  end

  def cellar_parameter_needed?(cellar)
    default_cellars = [
      Homebrew::DEFAULT_MACOS_CELLAR,
      Homebrew::DEFAULT_MACOS_ARM_CELLAR,
      Homebrew::DEFAULT_LINUX_CELLAR,
    ]
    cellar.present? && default_cellars.exclude?(cellar)
  end

  def generate_sha256_line(tag, digest, cellar, tag_column, digest_column)
    line = "sha256 "
    tag_column += line.length
    digest_column += line.length
    if cellar.is_a?(Symbol)
      line += "cellar: :#{cellar},"
    elsif cellar_parameter_needed?(cellar)
      line += %Q(cellar: "#{cellar}",)
    end
    line += " " * (tag_column - line.length)
    line += "#{tag}:"
    line += " " * (digest_column - line.length)
    %Q(#{line}"#{digest}")
  end

  def bottle_output(bottle)
    cellars = bottle.checksums.map do |checksum|
      cellar = checksum["cellar"]
      next unless cellar_parameter_needed? cellar

      case cellar
      when String
        %Q("#{cellar}")
      when Symbol
        ":#{cellar}"
      end
    end.compact
    tag_column = cellars.empty? ? 0 : "cellar: #{cellars.max_by(&:length)}, ".length

    tags = bottle.checksums.map { |checksum| checksum["tag"] }
    # Start where the tag ends, add the max length of the tag, add two for the `: `
    digest_column = tag_column + tags.max_by(&:length).length + 2

    sha256_lines = bottle.checksums.map do |checksum|
      generate_sha256_line(checksum["tag"], checksum["digest"], checksum["cellar"], tag_column, digest_column)
    end
    erb_binding = bottle.instance_eval { binding }
    erb_binding.local_variable_set(:sha256_lines, sha256_lines)
    erb = ERB.new BOTTLE_ERB
    erb.result(erb_binding).gsub(/^\s*$\n/, "")
  end

  def sudo_purge
    return unless ENV["HOMEBREW_BOTTLE_SUDO_PURGE"]

    system "/usr/bin/sudo", "--non-interactive", "/usr/sbin/purge"
  end

  def bottle_formula(f, args:)
    local_bottle_json = args.json? && f.local_bottle_path.present?

    unless local_bottle_json
      return ofail "Formula not installed or up-to-date: #{f.full_name}" unless f.latest_version_installed?
      return ofail "Formula was not installed with --build-bottle: #{f.full_name}" unless Utils::Bottles.built_as? f
    end

    tap = f.tap
    if tap.nil?
      return ofail "Formula not from core or any installed taps: #{f.full_name}" unless args.force_core_tap?

      tap = CoreTap.instance
    end

    if f.bottle_disabled?
      ofail "Formula has disabled bottle: #{f.full_name}"
      puts f.bottle_disable_reason
      return
    end

    return ofail "Formula has no stable version: #{f.full_name}" unless f.stable

    bottle_tag, rebuild = if local_bottle_json
      _, tag_string, rebuild_string = Utils::Bottles.extname_tag_rebuild(f.local_bottle_path.to_s)
      [tag_string.to_sym, rebuild_string.to_i]
    end

    bottle_tag = if bottle_tag
      Utils::Bottles::Tag.from_symbol(bottle_tag)
    else
      Utils::Bottles.tag
    end

    rebuild ||= if args.no_rebuild? || !tap
      0
    elsif args.keep_old?
      f.bottle_specification.rebuild
    else
      ohai "Determining #{f.full_name} bottle rebuild..."
      versions = FormulaVersions.new(f)
      rebuilds = versions.bottle_version_map("origin/master")[f.pkg_version]
      rebuilds.pop if rebuilds.last.to_i.positive?
      rebuilds.empty? ? 0 : rebuilds.max.to_i + 1
    end

    filename = Bottle::Filename.create(f, bottle_tag.to_sym, rebuild)
    local_filename = filename.to_s
    bottle_path = Pathname.pwd/filename

    tab = nil
    keg = nil

    tap_path = tap.path
    tap_git_revision = tap.git_head
    tap_git_remote = tap.remote

    root_url = args.root_url

    formulae_brew_sh_path = Utils::Analytics.formula_path

    relocatable = T.let(false, T::Boolean)
    skip_relocation = T.let(false, T::Boolean)

    prefix = HOMEBREW_PREFIX.to_s
    cellar = HOMEBREW_CELLAR.to_s

    if local_bottle_json
      bottle_path = f.local_bottle_path
      local_filename = bottle_path.basename.to_s

      tab_path = Utils::Bottles.receipt_path(bottle_path)
      raise "This bottle does not contain the file INSTALL_RECEIPT.json: #{bottle_path}" unless tab_path

      tab_json = Utils::Bottles.file_from_bottle(bottle_path, tab_path)
      tab = Tab.from_file_content(tab_json, tab_path)

      _, _, bottle_cellar = Formula[f.name].bottle_specification.checksum_for(bottle_tag, no_older_versions: true)
      relocatable = [:any, :any_skip_relocation].include?(bottle_cellar)
      skip_relocation = bottle_cellar == :any_skip_relocation

      prefix = bottle_tag.default_prefix
      cellar = bottle_tag.default_cellar
    else
      tar_filename = filename.to_s.sub(/.gz$/, "")
      tar_path = Pathname.pwd/tar_filename

      keg = Keg.new(f.prefix)
    end

    ohai "Bottling #{local_filename}..."

    formula_and_runtime_deps_names = [f.name] + f.runtime_dependencies.map(&:name)

    # this will be nil when using a local bottle
    keg&.lock do
      original_tab = nil
      changed_files = nil

      begin
        keg.delete_pyc_files!

        changed_files = keg.replace_locations_with_placeholders unless args.skip_relocation?

        Formula.clear_cache
        Keg.clear_cache
        Tab.clear_cache
        Dependency.clear_cache
        Requirement.clear_cache
        tab = Tab.for_keg(keg)
        original_tab = tab.dup
        tab.poured_from_bottle = false
        tab.HEAD = nil
        tab.time = nil
        tab.changed_files = changed_files.dup
        if args.only_json_tab?
          tab.changed_files.delete(Pathname.new(Tab::FILENAME))
          tab.tabfile.unlink
        else
          tab.write
        end

        keg.find do |file|
          # Set the times for reproducible bottles.
          if file.symlink?
            File.lutime(tab.source_modified_time, tab.source_modified_time, file)
          else
            file.utime(tab.source_modified_time, tab.source_modified_time)
          end
        end

        cd cellar do
          sudo_purge
          # Unset the owner/group for reproducible bottles.
          # Tar then gzip for reproducible bottles.
          safe_system "tar", "--create", "--numeric-owner", "--file", tar_path, "#{f.name}/#{f.pkg_version}"
          sudo_purge
          # Set more times for reproducible bottles.
          tar_path.utime(tab.source_modified_time, tab.source_modified_time)
          relocatable_tar_path = "#{f}-bottle.tar"
          mv tar_path, relocatable_tar_path
          # Use gzip, faster to compress than bzip2, faster to uncompress than bzip2
          # or an uncompressed tarball (and more bandwidth friendly).
          safe_system "gzip", "-f", relocatable_tar_path
          sudo_purge
          mv "#{relocatable_tar_path}.gz", bottle_path
        end

        ohai "Detecting if #{local_filename} is relocatable..." if bottle_path.size > 1 * 1024 * 1024

        prefix_check = if Homebrew.default_prefix?(prefix)
          File.join(prefix, "opt")
        else
          prefix
        end

        # Ignore matches to source code, which is not required at run time.
        # These matches may be caused by debugging symbols.
        ignores = [%r{/include/|\.(c|cc|cpp|h|hpp)$}]
        any_go_deps = f.deps.any? do |dep|
          dep.name =~ Version.formula_optionally_versioned_regex(:go)
        end
        if any_go_deps
          go_regex =
            Version.formula_optionally_versioned_regex(:go, full: false)
          ignores << %r{#{Regexp.escape(HOMEBREW_CELLAR)}/#{go_regex}/[\d.]+/libexec}
        end

        repository_reference = if HOMEBREW_PREFIX == HOMEBREW_REPOSITORY
          HOMEBREW_LIBRARY
        else
          HOMEBREW_REPOSITORY
        end.to_s
        if keg_contain?(repository_reference, keg, ignores, args: args)
          odie "Bottle contains non-relocatable reference to #{repository_reference}!"
        end

        relocatable = true
        if args.skip_relocation?
          skip_relocation = true
        else
          relocatable = false if keg_contain?(prefix_check, keg, ignores, formula_and_runtime_deps_names, args: args)
          relocatable = false if keg_contain?(cellar, keg, ignores, formula_and_runtime_deps_names, args: args)
          if keg_contain?(HOMEBREW_LIBRARY.to_s, keg, ignores, formula_and_runtime_deps_names, args: args)
            relocatable = false
          end
          if prefix != prefix_check
            relocatable = false if keg_contain_absolute_symlink_starting_with?(prefix, keg, args: args)
            relocatable = false if keg_contain?("#{prefix}/etc", keg, ignores, args: args)
            relocatable = false if keg_contain?("#{prefix}/var", keg, ignores, args: args)
            relocatable = false if keg_contain?("#{prefix}/share/vim", keg, ignores, args: args)
          end
          skip_relocation = relocatable && !keg.require_relocation?
        end
        puts if !relocatable && args.verbose?
      rescue Interrupt
        ignore_interrupts { bottle_path.unlink if bottle_path.exist? }
        raise
      ensure
        ignore_interrupts do
          original_tab&.write
          keg.replace_placeholders_with_locations changed_files unless args.skip_relocation?
        end
      end
    end

    bottle = BottleSpecification.new
    bottle.tap = tap
    bottle.root_url(root_url) if root_url
    bottle_cellar = if relocatable
      if skip_relocation
        :any_skip_relocation
      else
        :any
      end
    else
      cellar
    end
    bottle.rebuild rebuild
    sha256 = bottle_path.sha256
    bottle.sha256 cellar: bottle_cellar, bottle_tag.to_sym => sha256

    old_spec = f.bottle_specification
    if args.keep_old? && !old_spec.checksums.empty?
      mismatches = [:root_url, :prefix, :rebuild].reject do |key|
        old_spec.send(key) == bottle.send(key)
      end
      unless mismatches.empty?
        bottle_path.unlink if bottle_path.exist?

        mismatches.map! do |key|
          old_value = old_spec.send(key).inspect
          value = bottle.send(key).inspect
          "#{key}: old: #{old_value}, new: #{value}"
        end

        odie <<~EOS
          `--keep-old` was passed but there are changes in:
          #{mismatches.join("\n")}
        EOS
      end
    end

    output = bottle_output bottle

    puts "./#{local_filename}"
    puts output

    return unless args.json?

    bottle_filename = if bottle.root_url.match?(GitHubPackages::URL_REGEX)
      filename.github_packages
    else
      filename.url_encode
    end

    json = {
      f.full_name => {
        "formula" => {
          "name"             => f.name,
          "pkg_version"      => f.pkg_version.to_s,
          "path"             => f.path.to_s.delete_prefix("#{HOMEBREW_REPOSITORY}/"),
          "tap_git_path"     => f.path.to_s.delete_prefix("#{tap_path}/"),
          "tap_git_revision" => tap_git_revision,
          "tap_git_remote"   => tap_git_remote,
          # descriptions can contain emoji. sigh.
          "desc"             => f.desc.to_s.encode(
            Encoding.find("ASCII"),
            invalid: :replace, undef: :replace, replace: "",
          ).strip,
          "license"          => SPDX.license_expression_to_string(f.license),
          "homepage"         => f.homepage,
        },
        "bottle"  => {
          "root_url" => bottle.root_url,
          "prefix"   => bottle.prefix,
          "cellar"   => bottle_cellar.to_s,
          "rebuild"  => bottle.rebuild,
          "date"     => Pathname(local_filename).mtime.strftime("%F"),
          "tags"     => {
            bottle_tag.to_s => {
              "filename"              => bottle_filename,
              "local_filename"        => local_filename,
              "sha256"                => sha256,
              "formulae_brew_sh_path" => formulae_brew_sh_path,
              "tab"                   => tab.to_bottle_hash,
            },
          },
        },
      },
    }

    if bottle.root_url.match?(::Bintray::URL_REGEX) ||
       # TODO: given the naming: ideally the Internet Archive uploader wouldn't use this.
       bottle.root_url.start_with?("#{::Archive::URL_PREFIX}/")
      json[f.full_name]["bintray"] = {
        "package"    => Utils::Bottles::Bintray.package(f.name),
        "repository" => Utils::Bottles::Bintray.repository(tap),
      }
    end

    File.open(filename.json, "w") do |file|
      file.write JSON.pretty_generate json
    end
  end

  def parse_json_files(filenames)
    filenames.map do |filename|
      JSON.parse(File.read(filename))
    end
  end

  def merge_json_files(json_files)
    json_files.reduce({}) do |hash, json_file|
      json_file.each_value do |json_hash|
        json_bottle = json_hash["bottle"]
        cellar = json_bottle.delete("cellar")
        json_bottle["tags"].each_value do |json_platform|
          json_platform["cellar"] ||= cellar
        end
      end
      hash.deep_merge(json_file)
    end
  end

  def merge(args:)
    bottles_hash = merge_json_files(parse_json_files(args.named))

    # TODO: deduplicate --no-json bottles by:
    # 1. throwing away bottles for newer versions of macOS if their SHA256 is
    #    identical
    # 2. generating `all: $SHA256` bottles that can be used on macOS and Linux
    #    i.e. need to be `any_skip_relocation` and contain no ELF/MachO files.

    any_cellars = ["any", "any_skip_relocation"]
    bottles_hash.each do |formula_name, bottle_hash|
      ohai formula_name

      bottle = BottleSpecification.new
      bottle.root_url bottle_hash["bottle"]["root_url"]
      bottle.rebuild bottle_hash["bottle"]["rebuild"]
      bottle_hash["bottle"]["tags"].each do |tag, tag_hash|
        cellar = tag_hash["cellar"]
        cellar = cellar.to_sym if any_cellars.include?(cellar)
        sha256_hash = { cellar: cellar, tag.to_sym => tag_hash["sha256"] }
        bottle.sha256 sha256_hash
      end

      if args.write?
        Homebrew.install_bundler_gems!
        require "utils/ast"

        path = Pathname.new((HOMEBREW_REPOSITORY/bottle_hash["formula"]["path"]).to_s)
        formula = Formulary.factory(path)
        formula_ast = Utils::AST::FormulaAST.new(path.read)
        checksums = old_checksums(formula, formula_ast, bottle_hash, args: args)
        update_or_add = checksums.nil? ? "add" : "update"

        checksums&.each(&bottle.method(:sha256))
        output = bottle_output(bottle)
        puts output

        case update_or_add
        when "update"
          formula_ast.replace_bottle_block(output)
        when "add"
          formula_ast.add_bottle_block(output)
        end
        path.atomic_write(formula_ast.process)

        unless args.no_commit?
          Utils::Git.set_name_email!(committer: args.committer.blank?)
          Utils::Git.setup_gpg!

          if (committer = args.committer)
            committer = Utils.parse_author!(committer)
            ENV["GIT_COMMITTER_NAME"] = committer[:name]
            ENV["GIT_COMMITTER_EMAIL"] = committer[:email]
          end

          short_name = formula_name.split("/", -1).last
          pkg_version = bottle_hash["formula"]["pkg_version"]

          path.parent.cd do
            safe_system "git", "commit", "--no-edit", "--verbose",
                        "--message=#{short_name}: #{update_or_add} #{pkg_version} bottle.",
                        "--", path
          end
        end
      else
        puts bottle_output(bottle)
      end
    end
  end

  def merge_bottle_spec(old_keys, old_bottle_spec, new_bottle_hash)
    mismatches = []
    checksums = []

    new_values = {
      root_url: new_bottle_hash["root_url"],
      prefix:   new_bottle_hash["prefix"],
      rebuild:  new_bottle_hash["rebuild"],
    }

    skip_keys = [:sha256, :cellar]
    old_keys.each do |key|
      next if skip_keys.include?(key)

      old_value = old_bottle_spec.send(key).to_s
      new_value = new_values[key].to_s

      next if old_value.present? && new_value == old_value

      mismatches << "#{key}: old: #{old_value.inspect}, new: #{new_value.inspect}"
    end

    return [mismatches, checksums] if old_keys.exclude? :sha256

    old_bottle_spec.collector.each_key do |tag|
      old_checksum_hash = old_bottle_spec.collector[tag]
      old_hexdigest = old_checksum_hash[:checksum].hexdigest
      old_cellar = old_checksum_hash[:cellar]
      new_value = new_bottle_hash.dig("tags", tag.to_s)
      if new_value.present? && new_value["sha256"] != old_hexdigest
        mismatches << "sha256 #{tag}: old: #{old_hexdigest.inspect}, new: #{new_value["sha256"].inspect}"
      elsif new_value.present? && new_value["cellar"] != old_cellar.to_s
        mismatches << "cellar #{tag}: old: #{old_cellar.to_s.inspect}, new: #{new_value["cellar"].inspect}"
      else
        checksums << { cellar: old_cellar, tag => old_hexdigest }
      end
    end

    [mismatches, checksums]
  end

  def old_checksums(formula, formula_ast, bottle_hash, args:)
    bottle_node = formula_ast.bottle_block
    if bottle_node.nil?
      odie "`--keep-old` was passed but there was no existing bottle block!" if args.keep_old?
      return
    end
    return [] unless args.keep_old?

    old_keys = Utils::AST.body_children(bottle_node.body).map(&:method_name)
    old_bottle_spec = formula.bottle_specification
    mismatches, checksums = merge_bottle_spec(old_keys, old_bottle_spec, bottle_hash["bottle"])
    if mismatches.present?
      odie <<~EOS
        `--keep-old` was passed but there are changes in:
        #{mismatches.join("\n")}
      EOS
    end
    checksums
  end
end
