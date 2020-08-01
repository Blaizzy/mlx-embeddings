# frozen_string_literal: true

require "formula"
require "formula_creator"
require "missing_formula"
require "cli/parser"
require "utils/pypi"

module Homebrew
  module_function

  def create_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `create` [<options>] <URL>

        Generate a formula for the downloadable file at <URL> and open it in the editor.
        Homebrew will attempt to automatically derive the formula name and version, but
        if it fails, you'll have to make your own template. The `wget` formula serves as
        a simple example. For the complete API, see:
        <https://rubydoc.brew.sh/Formula>
      EOS
      switch "--autotools",
             description: "Create a basic template for an Autotools-style build."
      switch "--cmake",
             description: "Create a basic template for a CMake-style build."
      switch "--crystal",
             description: "Create a basic template for a Crystal build."
      switch "--go",
             description: "Create a basic template for a Go build."
      switch "--meson",
             description: "Create a basic template for a Meson-style build."
      switch "--node",
             description: "Create a basic template for a Node build."
      switch "--perl",
             description: "Create a basic template for a Perl build."
      switch "--python",
             description: "Create a basic template for a Python build."
      switch "--ruby",
             description: "Create a basic template for a Ruby build."
      switch "--rust",
             description: "Create a basic template for a Rust build."
      switch "--no-fetch",
             description: "Homebrew will not download <URL> to the cache and will thus not add its SHA-256 "\
                          "to the formula for you, nor will it check the GitHub API for GitHub projects "\
                          "(to fill out its description and homepage)."
      switch "--HEAD",
             description: "Indicate that <URL> points to the package's repository rather than a file."
      flag   "--set-name=",
             description: "Explicitly set the <name> of the new formula."
      flag   "--set-version=",
             description: "Explicitly set the <version> of the new formula."
      flag   "--set-license=",
             description: "Explicitly set the <license> of the new formula."
      flag   "--tap=",
             description: "Generate the new formula within the given tap, specified as <user>`/`<repo>."
      switch "-f", "--force",
             description: "Ignore errors for disallowed formula names and named that shadow aliases."

      conflicts "--autotools", "--cmake", "--crystal", "--go", "--meson", "--node", "--perl", "--python", "--rust"
      named 1
    end
  end

  # Create a formula from a tarball URL
  def create
    args = create_args.parse

    # Ensure that the cache exists so we can fetch the tarball
    HOMEBREW_CACHE.mkpath

    url = args.named.first # Pull the first (and only) url from ARGV

    version = args.set_version
    name = args.set_name
    license = args.set_license
    tap = args.tap

    fc = FormulaCreator.new(args)
    fc.name = name
    fc.version = version
    fc.license = license
    fc.tap = Tap.fetch(tap || "homebrew/core")
    raise TapUnavailableError, tap unless fc.tap.installed?

    fc.url = url

    fc.mode = if args.cmake?
      :cmake
    elsif args.autotools?
      :autotools
    elsif args.meson?
      :meson
    elsif args.crystal?
      :crystal
    elsif args.go?
      :go
    elsif args.node?
      :node
    elsif args.perl?
      :perl
    elsif args.python?
      :python
    elsif args.ruby?
      :ruby
    elsif args.rust?
      :rust
    end

    if fc.name.nil? || fc.name.strip.empty?
      stem = Pathname.new(url).stem
      print "Formula name [#{stem}]: "
      fc.name = __gets || stem
      fc.update_path
    end

    # Check for disallowed formula, or names that shadow aliases,
    # unless --force is specified.
    unless args.force?
      if reason = MissingFormula.disallowed_reason(fc.name)
        raise <<~EOS
          #{fc.name} is not allowed to be created.
          #{reason}
          If you really want to create this formula use --force.
        EOS
      end

      if Formula.aliases.include? fc.name
        realname = Formulary.canonical_name(fc.name)
        raise <<~EOS
          The formula #{realname} is already aliased to #{fc.name}
          Please check that you are not creating a duplicate.
          To force creation use --force.
        EOS
      end
    end

    fc.generate!

    PyPI.update_python_resources! Formula[fc.name], ignore_non_pypi_packages: true if args.python?

    puts "Please run `brew audit --new-formula #{fc.name}` before submitting, thanks."
    exec_editor fc.path
  end

  def __gets
    gots = $stdin.gets.chomp
    gots.empty? ? nil : gots
  end
end
