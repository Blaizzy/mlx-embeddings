# frozen_string_literal: true

require "cli/parser"

module Homebrew
  module_function

  def prof_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `prof` <command>

        Run Homebrew with the Ruby profiler, e.g. `brew prof readall`.
      EOS
    end
  end

  def prof
    prof_args.parse

    Homebrew.install_gem_setup_path! "ruby-prof", version: "0.18.0"
    FileUtils.mkdir_p "prof"
    brew_rb = (HOMEBREW_LIBRARY_PATH/"brew.rb").resolved_path
    safe_system "ruby-prof", "--printer=multi", "--file=prof", brew_rb, "--", *ARGV
  end
end
