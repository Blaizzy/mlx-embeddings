# typed: false
# frozen_string_literal: true

require "formulary"
require "cask/cask_loader"
require "cli/parser"

class String
  def f(*args)
    Formulary.factory(self, *args)
  end

  def c(config: nil)
    Cask::CaskLoader.load(self, config: config)
  end
end

class Symbol
  def f(*args)
    to_s.f(*args)
  end

  def c(config: nil)
    to_s.c(config: config)
  end
end

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def irb_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Enter the interactive Homebrew Ruby shell.
      EOS
      switch "--examples",
             description: "Show several examples."
      switch "--pry",
             env:         :pry,
             description: "Use Pry instead of IRB. Implied if `HOMEBREW_PRY` is set."
    end
  end

  def irb
    # work around IRB modifying ARGV.
    args = irb_args.parse(ARGV.dup.freeze)

    if args.examples?
      puts <<~EOS
        'v8'.f # => instance of the v8 formula
        :hub.f.latest_version_installed?
        :lua.f.methods - 1.methods
        :mpd.f.recursive_dependencies.reject(&:installed?)

        'vlc'.c # => instance of the vlc cask
        :tsh.c.livecheckable?
      EOS
      return
    end

    if args.pry?
      Homebrew.install_gem_setup_path! "pry"
      require "pry"
      Pry.config.prompt_name = "brew"
    else
      require "irb"
    end

    require "formula"
    require "keg"
    require "cask"

    ohai "Interactive Homebrew Shell", "Example commands available with: `brew irb --examples`"
    if args.pry?
      Pry.start
    else
      IRB.start
    end
  end
end
