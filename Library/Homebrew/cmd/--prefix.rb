# typed: false
# frozen_string_literal: true

require "cli/parser"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def __prefix_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Display Homebrew's install path. *Default:*

          - macOS Intel: `#{HOMEBREW_DEFAULT_PREFIX}`
          - macOS ARM: `#{HOMEBREW_MACOS_ARM_DEFAULT_PREFIX}`
          - Linux: `#{HOMEBREW_LINUX_DEFAULT_PREFIX}`

        If <formula> is provided, display the location in the Cellar where <formula>
        is or would be installed.
      EOS
      switch "--unbrewed",
             description: "List files in Homebrew's prefix not installed by Homebrew."

      named_args :formula
    end
  end

  def __prefix
    args = __prefix_args.parse

    if args.unbrewed?
      raise UsageError, "`--unbrewed` does not take a formula argument." unless args.no_named?

      list_unbrewed
    elsif args.no_named?
      puts HOMEBREW_PREFIX
    else
      puts args.named.to_resolved_formulae.map { |f|
        f.opt_prefix.exist? ? f.opt_prefix : f.latest_installed_prefix
      }
    end
  end

  UNBREWED_EXCLUDE_FILES = %w[.DS_Store].freeze
  UNBREWED_EXCLUDE_PATHS = %w[
    */.keepme
    .github/*
    bin/brew
    completions/zsh/_brew
    docs/*
    lib/gdk-pixbuf-2.0/*
    lib/gio/*
    lib/node_modules/*
    lib/python[23].[0-9]/*
    lib/pypy/*
    lib/pypy3/*
    lib/ruby/gems/[12].*
    lib/ruby/site_ruby/[12].*
    lib/ruby/vendor_ruby/[12].*
    manpages/brew.1
    share/pypy/*
    share/pypy3/*
    share/info/dir
    share/man/whatis
  ].freeze

  def list_unbrewed
    dirs  = HOMEBREW_PREFIX.subdirs.map { |dir| dir.basename.to_s }
    dirs -= %w[Library Cellar Caskroom .git]

    # Exclude cache, logs, and repository, if they are located under the prefix.
    [HOMEBREW_CACHE, HOMEBREW_LOGS, HOMEBREW_REPOSITORY].each do |dir|
      dirs.delete dir.relative_path_from(HOMEBREW_PREFIX).to_s
    end
    dirs.delete "etc"
    dirs.delete "var"

    arguments = dirs.sort + %w[-type f (]
    arguments.concat UNBREWED_EXCLUDE_FILES.flat_map { |f| %W[! -name #{f}] }
    arguments.concat UNBREWED_EXCLUDE_PATHS.flat_map { |d| %W[! -path #{d}] }
    arguments.concat %w[)]

    cd HOMEBREW_PREFIX
    safe_system "find", *arguments
  end
end
