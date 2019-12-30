# frozen_string_literal: true

require "stringio"
require "formula"
require "cli/parser"

module Homebrew
  module_function

  def unpack_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `unpack` [<options>] <formula>

        Unpack the source files for <formula> into subdirectories of the current
        working directory.
      EOS
      flag   "--destdir=",
             description: "Create subdirectories in the directory named by <path> instead."
      switch "--patch",
             description: "Patches for <formula> will be applied to the unpacked source."
      switch "-g", "--git",
             description: "Initialise a Git repository in the unpacked source. This is useful for creating "\
                          "patches for the software."
      switch :force
      switch :verbose
      switch :debug
      conflicts "--git", "--patch"
    end
  end

  def unpack
    unpack_args.parse

    formulae = Homebrew.args.formulae
    raise FormulaUnspecifiedError if formulae.empty?

    if dir = args.destdir
      unpack_dir = Pathname.new(dir).expand_path
      unpack_dir.mkpath
    else
      unpack_dir = Pathname.pwd
    end

    raise "Cannot write to #{unpack_dir}" unless unpack_dir.writable_real?

    formulae.each do |f|
      stage_dir = unpack_dir/"#{f.name}-#{f.version}"

      if stage_dir.exist?
        raise "Destination #{stage_dir} already exists!" unless args.force?

        rm_rf stage_dir
      end

      oh1 "Unpacking #{Formatter.identifier(f.full_name)} to: #{stage_dir}"

      ENV["VERBOSE"] = "1" # show messages about tar
      f.brew do
        f.patch if args.patch?
        cp_r getwd, stage_dir, preserve: true
      end
      ENV["VERBOSE"] = nil

      next unless args.git?

      ohai "Setting up git repository"
      cd stage_dir
      system "git", "init", "-q"
      system "git", "add", "-A"
      system "git", "commit", "-q", "-m", "brew-unpack"
    end
  end
end
