# frozen_string_literal: true

require "cli/parser"

module Homebrew
  module_function

  def ruby_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `ruby` (`-e` <text>|<file>)

        Run a Ruby instance with Homebrew's libraries loaded, e.g.
        `brew ruby -e "puts :gcc.f.deps"` or `brew ruby script.rb`.
      EOS
      flag "-r=",
           description: "Load a library using `require`."
      flag "-e=",
           description: "Execute the given text string as a script."
    end
  end

  def ruby
    args = ruby_args.parse

    ruby_sys_args = []
    ruby_sys_args << "-r#{args.r}" if args.r
    ruby_sys_args << "-e #{args.e}" if args.e
    ruby_sys_args += args.named

    begin
      safe_system RUBY_PATH,
                  ENV["HOMEBREW_RUBY_WARNINGS"],
                  "-I", $LOAD_PATH.join(File::PATH_SEPARATOR),
                  "-rglobal", "-rdev-cmd/irb",
                  *ruby_sys_args
    rescue ErrorDuringExecution => e
      exit e.status.exitstatus
    end
  end
end
