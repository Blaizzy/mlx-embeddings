# typed: false
# frozen_string_literal: true

require "cli/parser"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def prof_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `prof` [<command>]

        Run Homebrew with a Ruby profiler, e.g. `brew prof readall`.
      EOS
      switch "--stackprof",
             description: "Use `stackprof` instead of `ruby-prof` (the default)."

      named_args :command
    end
  end

  def prof
    args = prof_args.parse

    brew_rb = (HOMEBREW_LIBRARY_PATH/"brew.rb").resolved_path
    FileUtils.mkdir_p "prof"

    if args.stackprof?
      Homebrew.install_gem_setup_path! "stackprof"
      with_env HOMEBREW_STACKPROF: "1" do
        safe_system ENV["HOMEBREW_RUBY_PATH"], brew_rb, *args.named
      end
      output_filename = "prof/d3-flamegraph.html"
      safe_system "stackprof --d3-flamegraph prof/stackprof.dump > #{output_filename}"
    else
      Homebrew.install_gem_setup_path! "ruby-prof"
      output_filename = "prof/call_stack.html"
      safe_system "ruby-prof", "--printer=call_stack", "--file=#{output_filename}", brew_rb, "--", *args.named
    end

    exec_browser output_filename
  end
end
