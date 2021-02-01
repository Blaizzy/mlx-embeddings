# typed: false
# frozen_string_literal: true

require "metafiles"
require "formula"
require "cli/parser"
require "cask/cmd"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def list_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        List all installed formulae and casks.

        If <formula> is provided, summarise the paths within its current keg.
        If <cask> is provided, list its artifacts.
      EOS
      switch "--formula", "--formulae",
             description: "List only formulae, or treat all named arguments as formulae."
      switch "--cask", "--casks",
             description: "List only casks, or treat all named arguments as casks."
      switch "--unbrewed",
             description: "List files in Homebrew's prefix not installed by Homebrew.",
             replacement: "`brew --prefix --unbrewed`"
      switch "--full-name",
             description: "Print formulae with fully-qualified names. Unless `--full-name`, `--versions` "\
                          "or `--pinned` are passed, other options (i.e. `-1`, `-l`, `-r` and `-t`) are "\
                          "passed to `ls`(1) which produces the actual output."
      switch "--versions",
             description: "Show the version number for installed formulae, or only the specified "\
                          "formulae if <formula> are provided."
      switch "--multiple",
             depends_on:  "--versions",
             description: "Only show formulae with multiple versions installed."
      switch "--pinned",
             description: "List only pinned formulae, or only the specified (pinned) "\
                          "formulae if <formula> are provided. See also `pin`, `unpin`."
      # passed through to ls
      switch "-1",
             description: "Force output to be one entry per line. " \
                          "This is the default when output is not to a terminal."
      switch "-l",
             depends_on:  "--formula",
             description: "List formulae in long format."
      switch "-r",
             depends_on:  "--formula",
             description: "Reverse the order of the formulae sort to list the oldest entries first."
      switch "-t",
             depends_on:  "--formula",
             description: "Sort formulae by time modified, listing most recently modified first."

      conflicts "--formula", "--cask"
      conflicts "--full-name", "--versions"
      conflicts "--pinned", "--multiple"
      conflicts "--cask", "--multiple"
      ["--formula", "--cask", "--full-name", "--versions", "--pinned"].each do |flag|
        conflicts "--unbrewed", flag
      end
      ["-1", "-l", "-r", "-t"].each do |flag|
        conflicts "--unbrewed", flag
        conflicts "--versions", flag
        conflicts "--pinned", flag
      end
      ["--pinned", "-l", "-r", "-t"].each do |flag|
        conflicts "--full-name", flag
        conflicts "--cask", flag
      end

      named_args [:installed_formula, :installed_cask]
    end
  end

  def list
    args = list_args.parse

    # Unbrewed uses the PREFIX, which will exist
    # Things below use the CELLAR, which doesn't until the first formula is installed.
    unless HOMEBREW_CELLAR.exist?
      raise NoSuchKegError, args.named.first if args.named.present? && !args.cask?

      return
    end

    if args.full_name?
      unless args.cask?
        formula_names = args.no_named? ? Formula.installed : args.named.to_resolved_formulae
        full_formula_names = formula_names.map(&:full_name).sort(&tap_and_name_comparison)
        full_formula_names = Formatter.columns(full_formula_names) unless args.public_send(:'1?')
        puts full_formula_names if full_formula_names.present?
      end
      if args.cask? || (!args.formula? && args.no_named?)
        cask_names = if args.no_named?
          Cask::Caskroom.casks
        else
          args.named.to_formulae_and_casks(only: :cask, method: :resolve)
        end
        full_cask_names = cask_names.map(&:full_name).sort(&tap_and_name_comparison)
        full_cask_names = Formatter.columns(full_cask_names) unless args.public_send(:'1?')
        puts full_cask_names if full_cask_names.present?
      end
    elsif args.cask?
      list_casks(args: args)
    elsif args.pinned? || args.versions?
      filtered_list args: args
    elsif args.no_named?
      ENV["CLICOLOR"] = nil

      ls_args = []
      ls_args << "-1" if args.public_send(:'1?')
      ls_args << "-l" if args.l?
      ls_args << "-r" if args.r?
      ls_args << "-t" if args.t?

      safe_system "ls", *ls_args, HOMEBREW_CELLAR unless args.cask?
      list_casks(args: args) unless args.formula?
    elsif args.verbose? || !$stdout.tty?
      system_command! "find", args: args.named.to_kegs.map(&:to_s) + %w[-not -type d -print], print_stdout: true
    else
      args.named.to_kegs.each { |keg| PrettyListing.new keg }
    end
  end

  def filtered_list(args:)
    names = if args.no_named?
      Formula.racks
    else
      racks = args.named.map { |n| Formulary.to_rack(n) }
      racks.select do |rack|
        Homebrew.failed = true unless rack.exist?
        rack.exist?
      end
    end
    if args.pinned?
      pinned_versions = {}
      names.sort.each do |d|
        keg_pin = (HOMEBREW_PINNED_KEGS/d.basename.to_s)
        pinned_versions[d] = keg_pin.readlink.basename.to_s if keg_pin.exist? || keg_pin.symlink?
      end
      pinned_versions.each do |d, version|
        puts d.basename.to_s.concat(args.versions? ? " #{version}" : "")
      end
    else # --versions without --pinned
      names.sort.each do |d|
        versions = d.subdirs.map { |pn| pn.basename.to_s }
        next if args.multiple? && versions.length < 2

        puts "#{d.basename} #{versions * " "}"
      end
    end
  end

  def list_casks(args:)
    Cask::Cmd::List.list_casks(
      *args.named.to_casks,
      one:       args.public_send(:'1?'),
      full_name: args.full_name?,
      versions:  args.versions?,
      args:      args,
    )
  end
end

class PrettyListing
  def initialize(path)
    Pathname.new(path).children.sort_by { |p| p.to_s.downcase }.each do |pn|
      case pn.basename.to_s
      when "bin", "sbin"
        pn.find { |pnn| puts pnn unless pnn.directory? }
      when "lib"
        print_dir pn do |pnn|
          # dylibs have multiple symlinks and we don't care about them
          (pnn.extname == ".dylib" || pnn.extname == ".pc") && !pnn.symlink?
        end
      when ".brew"
        next # Ignore .brew
      else
        if pn.directory?
          if pn.symlink?
            puts "#{pn} -> #{pn.readlink}"
          else
            print_dir pn
          end
        elsif Metafiles.list?(pn.basename.to_s)
          puts pn
        end
      end
    end
  end

  def print_dir(root)
    dirs = []
    remaining_root_files = []
    other = ""

    root.children.sort.each do |pn|
      if pn.directory?
        dirs << pn
      elsif block_given? && yield(pn)
        puts pn
        other = "other "
      else
        remaining_root_files << pn unless pn.basename.to_s == ".DS_Store"
      end
    end

    dirs.each do |d|
      files = []
      d.find { |pn| files << pn unless pn.directory? }
      print_remaining_files files, d
    end

    print_remaining_files remaining_root_files, root, other
  end

  def print_remaining_files(files, root, other = "")
    if files.length == 1
      puts files
    elsif files.length > 1
      puts "#{root}/ (#{files.length} #{other}files)"
    end
  end
end
