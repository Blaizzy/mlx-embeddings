# typed: false
# frozen_string_literal: true

require "cli/parser"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def prof_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Run Homebrew with a Ruby profiler. For example, `brew prof readall`.

        *Note:* Not (yet) working on Apple Silicon.
      EOS
      switch "--stackprof",
             description: "Use `stackprof` instead of `ruby-prof` (the default)."

      named_args :command, min: 1
    end
  end

  def prof
    raise UsageError, "not (yet) working on Apple Silicon!" if Hardware::CPU.arm?

    args = prof_args.parse

    brew_rb = (HOMEBREW_LIBRARY_PATH/"brew.rb").resolved_path
    FileUtils.mkdir_p "prof"
    cmd = args.named.first
    raise UsageError, "#{cmd} is a Bash command!" if Commands.path(cmd).extname == ".sh"

    if args.stackprof?
      Homebrew.install_gem_setup_path! "stackprof"
      with_env HOMEBREW_STACKPROF: "1" do
        system ENV["HOMEBREW_RUBY_PATH"], brew_rb, *args.named
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
