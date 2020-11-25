# typed: true
# frozen_string_literal: true

module Homebrew
  module Compat
    attr_writer :args

    def args
      unless @printed_args_warning
        odisabled "Homebrew.args", "`args = <command>_args.parse` and pass `args` along the call chain"
      end

      @args ||= CLI::Args.new
    end
  end

  class << self
    prepend Compat
  end
end
