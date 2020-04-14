# frozen_string_literal: true

require "cli/parser"

module Homebrew
  module_function

  def cat_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `cat` <formula>

        Display the source of <formula>.
      EOS
      named :formula
    end
  end

  def cat
    cat_args.parse

    cd HOMEBREW_REPOSITORY
    pager = if Homebrew::EnvConfig.bat?
      "#{HOMEBREW_PREFIX}/bin/bat"
    else
      "cat"
    end
    safe_system pager, args.formulae_paths.first, *args.passthrough
  end
end
