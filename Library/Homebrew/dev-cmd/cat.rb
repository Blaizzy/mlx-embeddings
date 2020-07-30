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
    args = cat_args.parse

    cd HOMEBREW_REPOSITORY
    pager = if Homebrew::EnvConfig.bat?
      ENV["BAT_CONFIG_PATH"] = Homebrew::EnvConfig.bat_config_path
      "#{HOMEBREW_PREFIX}/bin/bat"
    else
      "cat"
    end
    safe_system pager, args.formulae_paths.first
  end
end
